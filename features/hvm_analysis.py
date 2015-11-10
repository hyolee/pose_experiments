import math
import numpy as np
import dldata.stimulus_sets.hvm as hvm
from dldata.metrics import utils
from dldata.metrics import classifier
import copy

from optparse import OptionParser
import cPickle
from archconvnets.convnet.api import assemble_feature_batches
import yamutils.fast as fast

dataset = hvm.HvMWithDiscfade()
meta = dataset.meta

def compute_perf(F, ntrain=None, ntest=5, num_splits=20, split_by='obj', var_levels=('V3', 'V6'), gridcv=False, attach_predictions=False, attach_models=False, reg_model_kwargs=None, justcat=False, model_type='svm.LinearSVC', model_kwargs=None):

    R = {}

    if model_kwargs is None:
        if gridcv:
            model_kwargs = {'GridSearchCV_params': {'C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4]}}
        else:
            model_kwargs = {'C':5e-3}

    basic_category_eval = {'npc_train': ntrain,
        'npc_test': ntest,
        'num_splits': num_splits,
        'npc_validate': 0,
        'metric_screen': 'classifier',
        'metric_labels': None,
        'metric_kwargs': {'model_type': model_type,
                          'model_kwargs': model_kwargs
                         },
        'labelfunc': 'category',
        'train_q': {'var': list(var_levels)},
        'test_q': {'var': list(var_levels)},
        'split_by': split_by}

    R['basic_category_result'] = utils.compute_metric(F, dataset, basic_category_eval, attach_models=attach_models, attach_predictions=attach_predictions)
    print('...finished categorization')
    if justcat:
        return R

    masks = dataset.pixel_masks

    volumes = masks.sum(1).sum(1)

    def get_axis_bb(x):
        nzx, nzy = x.nonzero()
        if len(nzx):
            return (nzy.min(), nzx.min(), nzy.max(), nzx.max())
        else:
            return (-128, -128, -128, -128)

    axis_bb = np.array(map(get_axis_bb, masks))

    get_axis_bb_ctr = lambda x: ((x[1] + x[3])/2, (x[0] + x[2])/2)

    axis_bb_ctr = np.array(map(get_axis_bb_ctr, axis_bb))

    axis_bb_sx = np.abs(axis_bb[:, 2] - axis_bb[:, 0])
    axis_bb_sy = np.abs(axis_bb[:, 3] - axis_bb[:, 1])
    axis_bb_area = axis_bb_sx * axis_bb_sy

    axis_bb_sx = np.abs(axis_bb[:, 2] - axis_bb[:, 0])
    axis_bb_sy = np.abs(axis_bb[:, 3] - axis_bb[:, 1])
    axis_bb_area = axis_bb_sx * axis_bb_sy

    borders = dataset.pixel_borders(thickness=1)
    area_bb = np.array(map(hvm.get_best_box, borders))
    area_bb[3818] = -128

    def line(a1, a2):
        if a1[0] != a2[0] and a1[1] != a2[1]:
            m = (a2[1] - a1[1]) / float(a2[0] - a1[0])
            b = a1[1] - m * a1[0]
        elif a1[1] == a2[1]:
            m = np.inf
            b = np.nan
        elif a1[0] == a2[0]:
            m = 0
            b = a1[0]
        return m, b

    def intersection(a):
        a1, a2, a3, a4 = a
        m1, b1 = line(a1, a4)
        m2, b2 = line(a2, a3)
        if not np.isinf(m1):
            if not np.isinf(m2):
                ix = (b2 - b1)/(m1 - m2)
            else:
                ix = a3[1]
            iy = m1 * ix + b1
        else:
            ix = a1[1]
            iy = m2 * ix + b2
        return ix, iy

    area_bb_ctr = np.array(map(intersection, area_bb))
    dist = lambda _b1, _b2: math.sqrt((_b1[0] - _b2[0])**2 + (_b1[1] - _b2[1])**2)
    def sminmax(a):
        a1, a2, a3, a4 = a
        s1 = dist(a1, a2)
        s2 = dist(a1, a3)
        smin = min(s1, s2)
        smax = max(s1, s2)
        return smin, smax

    area_bb_minmax = np.array(map(sminmax, area_bb))
    area_bb_area = area_bb_minmax[:, 0] * area_bb_minmax[:, 1]

    dist = lambda _b1, _b2: np.sqrt((_b1[0] - _b2[0])**2 + (_b1[1] - _b2[1])**2)

    def get_major_axis(a):
        a1, a2, a3, a4 = a
        s1 = dist(a1, a2)
        s2 = dist(a1, a3)
        if s1 > s2:
            m11 = a1
            m12 = a3
            m21 = a2
            m22 = a4
        else:
            m11 = a1
            m12 = a2
            m21 = a3
            m22 = a4

        mj1 = (m11 + m12)/2
        mj2 = (m21 + m22)/2
        return (mj1, mj2)

    def nonan(f):
        f[np.isnan(f)] = 0
        return f

    area_bb_major_axes = np.array(map(get_major_axis, area_bb))
    sin_maj = nonan((area_bb_major_axes[:, 0, 1] - area_bb_major_axes[:, 1, 1]) / dist(area_bb_major_axes[:, 0].T, area_bb_major_axes[:, 1].T))
    cos_maj = nonan((area_bb_major_axes[:, 0, 0] - area_bb_major_axes[:, 1, 0]) / dist(area_bb_major_axes[:, 0].T, area_bb_major_axes[:, 1].T))

    if reg_model_kwargs is None:
        reg_model_kwargs = {'alphas': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]}

    def make_spec(f):
        x = {'npc_train': ntrain,
        'npc_test': ntest,
        'num_splits': num_splits,
        'npc_validate': 0,
        'metric_screen': 'regression',
        'metric_labels': None,
        'metric_kwargs': {'model_type': 'linear_model.RidgeCV',
                          'model_kwargs': reg_model_kwargs},
        'train_q': {'var': list(var_levels)},
        'test_q': {'var': list(var_levels)},
        'split_by': split_by}

        x['labelfunc'] = f
        return x

    volume_spec = make_spec(lambda x: (volumes, None))
    R['volume_result'] = utils.compute_metric(F, dataset, volume_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished volume')

    perimeters = borders.sum(1).sum(1)
    perimeter_spec = make_spec(lambda x: (perimeters, None))
    R['perimeter_result'] = utils.compute_metric(F, dataset, perimeter_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished perimeter')

    axis_bb_sx_spec = make_spec(lambda x: (axis_bb_sx, None))
    axis_bb_sy_spec = make_spec(lambda x: (axis_bb_sy, None))
    axis_bb_asp_spec = make_spec(lambda x: (axis_bb_sx / np.maximum(axis_bb_sy, 1e-5), None))
    axis_bb_area_spec = make_spec(lambda x: (axis_bb_area, None))

    R['axis_bb_sx_result'] = utils.compute_metric(F, dataset, axis_bb_sx_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_sx')
    R['axis_bb_sy_result'] = utils.compute_metric(F, dataset, axis_bb_sy_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_sy')
    R['axis_bb_asp_result'] = utils.compute_metric(F, dataset, axis_bb_asp_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_asp')
    R['axis_bb_area_result'] = utils.compute_metric(F, dataset, axis_bb_area_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_area')

    area_bb_smin_spec = make_spec(lambda x: (area_bb_minmax[:, 0], None))
    area_bb_smax_spec = make_spec(lambda x: (area_bb_minmax[:, 1], None))

    area_bb_asps = area_bb_minmax[:, 1]/np.maximum(area_bb_minmax[:, 0], 1e-5)
    area_bb_asp_spec = make_spec(lambda x: (area_bb_asps, None))
    area_bb_area_spec = make_spec(lambda x: (area_bb_area, None))

    sinmajs = np.abs(sin_maj)
    sinmaj_spec = make_spec(lambda x: (sinmajs, None))

    R['area_bb_smin_result'] = utils.compute_metric(F, dataset, area_bb_smin_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_smin')
    R['area_bb_smax_result'] = utils.compute_metric(F, dataset, area_bb_smax_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_smax')
    R['area_bb_asp_result'] = utils.compute_metric(F, dataset, area_bb_asp_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_asp')
    R['area_bb_area_result'] = utils.compute_metric(F, dataset, area_bb_area_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_area')
    R['sinmaj_result'] = utils.compute_metric(F, dataset, sinmaj_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... sinmaj')

    #position
    posx_spec = make_spec(lambda x: (x['ty'], None))
    posy_spec = make_spec(lambda x: (x['tz'], None))
    R['posx_result'] = utils.compute_metric(F, dataset, posx_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    R['posy_result'] = utils.compute_metric(F, dataset, posy_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)


    size_spec = make_spec(lambda x: (x ['s'], None))
    R['size_result'] = utils.compute_metric(F, dataset, size_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... size')

    masksv0 = masks[meta['var'] == 'V0'][::10]
    objs = meta['obj'][meta['var'] == 'V0'][::10]
    A = np.array([_x.nonzero()[0][[0, -1]] if _x.sum() > 0 else (0, 0) for _x in masksv0.sum(1) ])
    B = np.array([_x.nonzero()[0][[0, -1]] if _x.sum() > 0 else (0, 0) for _x in masksv0.sum(2) ])
    base_size_factors = dict(zip( objs, np.maximum(A[:, 1] - A[:, 0], B[:, 1] - B[:, 0])))
    size_factors = np.zeros(len(meta))
    for o in objs:
        inds = [meta['obj'] == o]
        size_factors[inds] = base_size_factors[o] * .01
    scaled_size  = size_factors * meta['s']
    scaled_size_spec = make_spec(lambda x: (scaled_size, None))
    R['scaled_size_result'] = utils.compute_metric(F, dataset, scaled_size_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... scaled size')

    rxy_spec = make_spec(lambda x: (x['rxy_semantic'], None))
    rxz_spec = make_spec(lambda x: (x['rxz_semantic'], None))
    ryz_spec = make_spec(lambda x: (x['ryz_semantic'], None))
    R['rxy_result'] = utils.compute_metric(F, dataset, rxy_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    R['rxz_result'] = utils.compute_metric(F, dataset, rxz_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    R['ryz_result'] = utils.compute_metric(F, dataset, ryz_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... rotations')

    return R

def post_process_neural_regression_msplit(dataset, result, spec, n_jobs=1, splits=None):
    name = spec[0]
    specval = spec[1]
    assert name[2] in ['IT_regression', 'V4_regression', 'ITc_regression', 'ITt_regression'], name
    if name[2] == 'IT_regression':
        units = dataset.IT_NEURONS
    elif name[2] == 'ITc_regression':
        units = hvm.mappings.LST_IT_Chabo
    elif name[2] == 'ITt_regression':
        units = hvm.mappings.LST_IT_Tito
    else:
        units = dataset.V4_NEURONS

    units = np.array(units)
    if not splits:
        splits, validations = utils.get_splits_from_eval_config(specval, dataset)

    sarrays = []
    for s_ind, s in enumerate(splits):
        ne = dataset.noise_estimate(s['test'], units=units, n_jobs=n_jobs, cache=True)
        farray = np.asarray(result['split_results'][s_ind]['test_multi_rsquared'])
        sarray = farray / ne[0]**2
        sarrays.append(sarray)
    sarrays = np.asarray(sarrays)
    result['noise_corrected_multi_rsquared_array_loss'] = (1 - sarrays).mean(0)
    result['noise_corrected_multi_rsquared_array_loss_stderror'] = (1 - sarrays).std(0)
    result['noise_corrected_multi_rsquared_loss'] = np.median(1 - sarrays, 1).mean()
    result['noise_corrected_multi_rsquared_loss_stderror'] = np.median(1 - sarrays, 1).std()

def compute_nfit(features): 
    IT_feats = dataset.neuronal_features[:, dataset.IT_NEURONS]
    V4_feats = dataset.neuronal_features[:, dataset.V4_NEURONS]
    spec_IT_reg = {'npc_train': 70,
               'npc_test': 10,
               'num_splits': 5,
               'npc_validate': 0,
               'metric_screen': 'regression',
               'metric_labels': None,
               'metric_kwargs': {'model_type': 'pls.PLSRegression',
                                 'model_kwargs': {'n_components':25}},
                'labelfunc': lambda x: (IT_feats, None),
                'train_q': {'var':['V3', 'V6']},
                'test_q': {'var':['V3', 'V6']},
                'split_by': 'obj'}
    spec_V4_reg = copy.deepcopy(spec_IT_reg)
    spec_V4_reg['labelfunc'] = lambda  x: (V4_feats, None)
    
    result_IT = utils.compute_metric(features, dataset, spec_IT_reg)
    print('... IT fit')
    result_V4 = utils.compute_metric(features, dataset, spec_V4_reg)
    print('... V4 fit')    

    # noise correction
    espec = (('all','','IT_regression'), spec_IT_reg)
    post_process_neural_regression_msplit(dataset, result_IT, espec, n_jobs=1)
    espec = (('all','','V4_regression'), spec_V4_reg)
    post_process_neural_regression_msplit(dataset, result_V4, espec, n_jobs=1)

    return {'IT': result_IT['noise_corrected_multi_rsquared_loss'], 'V4': result_V4['noise_corrected_multi_rsquared_loss']}

def main():
    print "parsing options..."
    parser = OptionParser()
    parser.add_option("-f", "--feature-dir", dest="feature_dir")
    parser.add_option("-n", "--Nsubsample", type="int", dest="Nsub", default=None)
    parser.add_option("-m", "--metric", dest="metric", default='pn')
    (options, args) = parser.parse_args()
    
    feature_dir = options.feature_dir
    Nsub = options.Nsub
    metric = options.metric
    
    fds = feature_dir.split(',')
    
    features = cPickle.load(open(fds[0]))
    #features = assemble_feature_batches(fds[0])
    perm = np.random.RandomState(0).permutation(features.shape[0])
    pinv = fast.perminverse(perm)
    features = features[pinv] 
    if len(fds) > 1:
        features2 = cPickle.load(open(fds[1]))
#        features2 = assemble_feature_batches(fds[1])
        perm = np.random.RandomState(0).permutation(features2.shape[0])
        pinv = fast.perminverse(perm)
        features = np.append( features, features2[pinv], axis=1 )
    if Nsub is not None:
        features = features[pinv][:,perm[:Nsub]]
    
    RR = {}
    if 'p' in metric:
        print "Compute hvm performance..."
        RR['p'] = compute_perf(features, var_levels=['V6'])
    if 'n' in metric:
        print "Compute hvm neural fit..."
        RR['n'] = compute_nfit(features)
    
    cPickle.dump( RR, open(fds[0] + "result_hvm_" + str(Nsub) + ".p", "wb") )

if __name__ == "__main__":
    main()

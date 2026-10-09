from copy import deepcopy
from evaluation.fast_new_training import stratified_ids, validate_score


def test_subset_is_fixed_count_stratified_and_does_not_use_targets():
    rows = [{'id':str(i),'language':['en','de'][i%2],
             '_new_dataset':{'source_id':['documents','games'][i%3 == 0]},
             'target_probs':[float(i%2),float((i+1)%2)]} for i in range(100)]
    lengths = {str(i):[200,1000,4000,20000,35000][i%5] for i in range(100)}
    first,cells = stratified_ids(rows,lengths,32,seed=42)
    assert len(first) == len(set(first)) == 32
    altered = deepcopy(rows)
    for row in altered:
        row['target_probs'] = [0.3,0.7]
    second,_ = stratified_ids(list(reversed(altered)),lengths,32,seed=42)
    assert first == second
    assert len(cells) == 20
    chosen = [r for r in rows if r['id'] in first]
    assert {r['_new_dataset']['source_id'] for r in chosen} == {'documents','games'}
    assert {r['language'] for r in chosen} == {'en','de'}
    assert {lengths[r['id']] for r in chosen} == {200,1000,4000,20000,35000}
    third,_ = stratified_ids(rows,lengths,32,seed=43)
    assert third != first


def test_partial_or_nonfinite_score_is_rejected():
    import pytest
    score = {'questions':128,'soft_cross_entropy':1.0,'kl_pq':0.2,'categorical_brier_sum':0.3,'argmax_agreement':0.5}
    validate_score(score,128)
    with pytest.raises(ValueError,match='Partial'):
        validate_score({**score,'questions':127},128)
    with pytest.raises(ValueError,match='nonfinite'):
        validate_score({**score,'soft_cross_entropy':float('nan')},128)


def test_source_quotas_follow_population_instead_of_language_cell_count():
    rows = [{'id':str(i),'language':str(i) if i<100 else 'en',
             '_new_dataset':{'source_id':'old' if i<100 else 'games'}} for i in range(1000)]
    ids,_ = stratified_ids(rows,{str(i):200 for i in range(1000)},128)
    assert sum(int(i)<100 for i in ids) == 13
    assert sum(int(i)>=100 for i in ids) == 115

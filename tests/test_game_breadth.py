from fractions import Fraction
import json
from pathlib import Path
import random
import pytest

from dataset_factory.game_data.breadth import (
    dice_sample, mastermind_key, mastermind_sample, make_row, recompute, split_rows, generate,
)
from dataset_factory.game_data.mastermind import Mastermind
from dataset_factory.game_data.pilot import generate as generate_pilot


def test_known_law_forecast_is_not_unknown_realized_fact_idk():
    # Reusing the seed gives exactly the same law before query branching.
    forecast = dice_sample(random.Random(17), 0)
    realized = dice_sample(random.Random(17), 4)
    assert json.loads(forecast['context'])['weights'] == json.loads(realized['context'])['weights']
    assert forecast['answerability_status'] == 'answerable'
    assert forecast['target_distribution'][-1] == 0
    assert realized['answerability_status'] == 'unanswerable'
    assert realized['target_distribution'][-1] == 1
    assert forecast['id'] != realized['id']
    assert forecast['game_metadata']['split_group_keys'] == realized['game_metadata']['split_group_keys']
    recompute(forecast); recompute(realized)


def test_mastermind_canonical_identity_covers_labels_positions_history_and_candidates():
    state = Mastermind(4, 3, (((0,1,2),1,1),))
    renamed = Mastermind(4, 3, (((2,3,1),1,1),))
    assert mastermind_key(state) == mastermind_key(renamed)
    rotated = Mastermind(4, 3, (((2,0,1),1,1),))
    assert mastermind_key(state, {'position':0}) == mastermind_key(rotated, {'position':1})
    candidates = ((0,0,3),(1,2,3))
    assert mastermind_key(state, {'candidates':candidates}) == mastermind_key(state, {'candidates':tuple(reversed(candidates))})
    code = (0,1,2)
    renamed_code = (2,3,1)
    assert mastermind_key(state, {'guess':code}) == mastermind_key(renamed, {'guess':renamed_code})


def test_posterior_targets_use_public_history_only_and_survive_option_alignment():
    row = next(row for i in range(30) if (row := mastermind_sample(random.Random(17+i), i)) is not None)
    observation = json.loads(row['context'])
    assert not ({'secret','hidden_code','actual_code'} & observation.keys())
    recompute(row)
    positive = next(i for i,p in enumerate(row['target_distribution']) if p > 0)
    zero = len(row['target_distribution']) - 1
    row['target_distribution'][positive],row['target_distribution'][zero] = row['target_distribution'][zero],row['target_distribution'][positive]
    with pytest.raises(ValueError): recompute(row)


def test_prior_grid_relations_preserve_or_quarantine_new_pushing_family():
    walls=[int(r==0 or c==0 or r==4 or c==4) for r in range(5) for c in range(5)]
    def prior(split,index):
        return {'id':f'prior-{index}','topic':'grid_navigation','split':split,'problem_group_id':f'old-{index}',
                'context':json.dumps({'size':5,'walls':walls}), 'game_metadata':{'state_family_id':f'old-state-{index}'}}
    def new():
        return {'id':'new','topic':'sokoban','split':'development','problem_group_id':'pending',
                'context':json.dumps({'width':5,'height':5,'walls':walls}),
                'game_metadata':{'state_family_id':'new-state','split_group_keys':['sokoban-layout']}}
    first=new(); split_rows([prior('train',0)],[first],17)
    assert first['split']=='train'
    second=new(); old=[prior('train',0),prior('development',1)]; original=json.dumps(old)
    split_rows(old,[second],17)
    assert second['split']=='quarantine' and json.dumps(old)==original


def test_small_balanced_run_verifies_all_rows_and_keeps_prior_files_frozen(tmp_path):
    prior=tmp_path/'pilot'; generate_pilot(prior,tmp_path/'pilot-status.json',per_game=2,seed=17)
    output=tmp_path/'breadth'; status=tmp_path/'status.json'
    result=generate(output,status,per_family=5,seed=17,max_seconds=60,prior_dirs=(prior,))
    assert result['state']=='complete'
    assert result['generated_new_originals']==15
    assert result['accepted_by_family']=={'dice_decisions':5,'mastermind':5,'sokoban':5}
    assert result['verification']['all_targets_recomputed']
    assert not result['gpu_used'] and result['data_authoring_api_calls']==0
    assert result['balanced_sampling']['training_family_weights']=={family:Fraction(1,3) for family in ('dice_decisions','mastermind','sokoban')}
    rows=[]
    for record in result['files']:
        rows.extend(json.loads(line) for line in Path(record['path']).read_text().splitlines())
    assert len({row['id'] for row in rows})==15
    for row in rows: recompute(row)
    assert {r['problem_group_id'] for r in rows if r['split']=='train'}.isdisjoint({r['problem_group_id'] for r in rows if r['split']=='development'})
    with pytest.raises(ValueError): generate(output,status,per_family=5,prior_dirs=(prior,))


def test_repeated_public_feedback_does_not_count_as_a_new_question():
    observation = ((0,1,2),1,1)
    state = Mastermind(4,3,(observation,))
    repeated = Mastermind(4,3,(observation,observation))
    assert state.posterior() == repeated.posterior()
    assert mastermind_key(state) == mastermind_key(repeated)
    for query in ({'position':0},{'guess':(0,2,1)},{'candidates':((0,0,0),(1,2,3))}):
        assert mastermind_key(state,query) == mastermind_key(repeated,query)


def test_quality_publication_excludes_repeated_history_variant_with_unchanged_labels(tmp_path):
    import copy
    import hashlib
    from dataset_factory.game_data.breadth_quality import finalize
    import dataset_factory.game_data.breadth as module

    row = dice_sample(random.Random(17), 0)
    for index in range(100):
        candidate = mastermind_sample(random.Random(index), 0)
        if candidate is not None and len(json.loads(candidate['context'])['history']) == 1:
            row = candidate
            break
    assert row['topic'] == 'mastermind'
    row['split'] = 'train'
    repeated = copy.deepcopy(row)
    repeated['id'] = 'historical-redundant-row'
    observation = json.loads(repeated['context'])
    observation['history'] += observation['history'][:1]
    repeated['context'] = json.dumps(observation, sort_keys=True)
    directory = tmp_path / 'quality'; directory.mkdir()
    path = directory / 'train.jsonl'
    text = ''.join(json.dumps(item) + '\n' for item in (row,repeated)); path.write_text(text)
    generator = Path(module.__file__).resolve(); snapshot = tmp_path / 'generator.py'; snapshot.write_bytes(generator.read_bytes())
    manifest = {'generated_new_originals':2,'files':[{'path':str(path),'rows':2,'sha256':hashlib.sha256(text.encode()).hexdigest()}],
                'prior':[],'diversity':{},'verification':{},'limitations':[],
                'provenance':{'source_sha256':{str(generator):hashlib.sha256(generator.read_bytes()).hexdigest()}}}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    result = finalize(directory,tmp_path/'status.json',snapshot)
    assert result['accepted_originals'] == result['generated_new_originals'] == 1
    assert result['generated_candidate_rows'] == 2
    assert result['quality_excluded_originals'] == 1
    saved = json.loads((directory/'train.jsonl').read_text())
    assert saved['target_distribution'] == row['target_distribution']
    recompute(saved)
    with pytest.raises(ValueError,match='already published'):
        finalize(directory,tmp_path/'status.json',snapshot)

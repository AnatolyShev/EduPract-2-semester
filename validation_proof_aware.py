"""Статически доказуемые ошибки и эмпирические свидетельства достижимости.

Запуск: python validation_proof_aware.py
Зависимости: pydantic >= 2. Не использует LLM, GPU или интернет.

Статусы:
    PROVEN_INVALID             — нарушение контракта либо достаточное условие невозможности.
    WITNESSED_OFFENSIVE_WIN    — конкретная выигрышная партия найдена (НЕ универсальная корректность).
    INCONCLUSIVE              — ни доказательства ошибки, ни выигрышного примера нет.

Внимание: сертификат верхней границы доказывает недостижимость только ПОБЕДЫ
ПОСРЕДСТВОМ НАНЕСЕНИЯ УРОНА СОПЕРНИКУ и в рамках формально заданного симулятора.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, ValidationError, model_validator


class CardRule(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1, max_length=50)
    cost: int = Field(ge=0, le=10)
    effect: Literal['damage', 'heal', 'draw']
    value: int = Field(ge=0, le=20)
    target: Literal['enemy', 'self']


class GameRules(BaseModel):
    model_config = ConfigDict(extra='forbid')
    game_name: str
    max_rounds: int = Field(ge=1, le=50)
    starting_hp: int = Field(ge=1, le=100)
    starting_hand_size: int = Field(ge=1, le=7)
    deck_size: int = Field(ge=4, le=60)
    mana_cap: int = Field(ge=1, le=10)
    cards: list[CardRule] = Field(min_length=2, max_length=15)
    win_condition: Literal['hp_zero'] = 'hp_zero'

    @model_validator(mode='after')
    def check_deck(self):
        if self.starting_hand_size > self.deck_size:
            raise ValueError('starting_hand_size must not exceed deck_size')
        return self


BASE = {
    'game_name': 'Fixed Card Duel',
    'max_rounds': 12, 'starting_hp': 10, 'mana_cap': 3,
    'starting_hand_size': 3, 'deck_size': 12, 'win_condition': 'hp_zero',
    'cards': [
        {'name': 'strike', 'cost': 1, 'effect': 'damage', 'value': 3, 'target': 'enemy'},
        {'name': 'blast', 'cost': 2, 'effect': 'damage', 'value': 4, 'target': 'enemy'},
        {'name': 'restore', 'cost': 2, 'effect': 'heal', 'value': 2, 'target': 'self'},
        {'name': 'insight', 'cost': 1, 'effect': 'draw', 'value': 1, 'target': 'self'},
    ],
}


def scenarios():
    """14 исходных тестов + 1 дополнительный корректный с ловушкой для Greedy."""
    cases = [('base', deepcopy(BASE), True, 'исходные корректные правила')]
    edits = [
        ('cheaper_attacks', True, 'уменьшена стоимость атак', lambda r: [c.update(cost=0) for c in r['cards'] if c['effect'] == 'damage']),
        ('stronger_attacks', True, 'увеличен урон атак', lambda r: [c.update(value=5) for c in r['cards'] if c['effect'] == 'damage']),
        ('longer_game', True, 'увеличено число раундов', lambda r: r.update(max_rounds=18)),
        ('no_healing', True, 'лечение заменено корректной атакой', lambda r: r['cards'][2].update(effect='damage', target='enemy', value=2)),
        ('weak_but_reachable', True, 'урон = 2, достижимость возможна', lambda r: [c.update(value=2) for c in r['cards'] if c['effect'] == 'damage']),
        ('no_attacks', False, 'нет атак по противнику', lambda r: [c.update(effect='heal', target='self', value=2) for c in r['cards'] if c['effect'] == 'damage']),
        ('zero_damage', False, 'все атаки имеют нулевой урон', lambda r: [c.update(value=0) for c in r['cards'] if c['effect'] == 'damage']),
        ('unaffordable_attacks', False, 'стоимость атак превышает запас маны', lambda r: [c.update(cost=4) for c in r['cards'] if c['effect'] == 'damage']),
        ('short_horizon', False, 'недостаточно раундов для 10 HP', lambda r: r.update(max_rounds=2)),
        ('weak_and_short', False, 'слабый урон и короткий горизонт', lambda r: (r.update(max_rounds=4), [c.update(value=1) for c in r['cards'] if c['effect'] == 'damage'])),
        ('attacks_self', False, 'damage неверно направлен на self', lambda r: [c.update(target='self') for c in r['cards'] if c['effect'] == 'damage']),
        ('heals_enemy', False, 'heal неверно направлен на enemy', lambda r: r['cards'][2].update(target='enemy')),
        ('mana_bottleneck', False, 'за 5 раундов не более одной атаки за 3 маны', lambda r: (r.update(max_rounds=5), [c.update(cost=3) for c in r['cards'] if c['effect'] == 'damage'])),
        ('strategy_trap', True, 'можно ждать три раунда и сыграть blast, но Greedy сразу расходует ману на strike',
         lambda r: (r.update(max_rounds=3, deck_size=4, starting_hand_size=4),
                    r['cards'][0].update(value=1), r['cards'][1].update(cost=3, value=10))),
    ]
    for name, valid, description, change in edits:
        r = deepcopy(BASE)
        change(r)
        cases.append((name, r, valid, description))
    return cases


def structural_validation(raw):
    try:
        return GameRules.model_validate(raw), []
    except ValidationError as exc:
        return None, [x['msg'] for x in exc.errors()]


def deck_template(rules: GameRules):
    return (rules.cards * math.ceil(rules.deck_size / len(rules.cards)))[:rules.deck_size]


def contract_errors(rules: GameRules):
    """Нарушения явно объявленного контракта, а не экспертного 'баланса'."""
    errors = []
    if len({c.name for c in rules.cards}) != len(rules.cards):
        errors.append('CONTRACT_DUPLICATE_NAMES: имена карт должны быть уникальными')
    for c in rules.cards:
        target = 'enemy' if c.effect == 'damage' else 'self'
        if c.target != target:
            errors.append(f'CONTRACT_TARGET: {c.name} ({c.effect}) требует target={target}')
        if c.effect in ('damage', 'heal') and c.value == 0:
            errors.append(f'CONTRACT_ZERO_EFFECT: {c.name} имеет нулевую силу {c.effect}')
        if c.effect == 'draw' and c.value != 1:
            errors.append(f'CONTRACT_DRAW: {c.name} должен иметь value=1')
    return errors


def optimistic_damage_bound(rules: GameRules):
    """ГАРАНТИРОВАННАЯ верхняя граница суммарного урона за R раундов.

    Для игрока стартовая мана=0, прирост <=1 за раунд, значит:
    - суммарно потрачено не более R маны;
    - сыграно не более R карт (одно действие за ход);
    - экземпляр каждой карты не может использоваться повторно;
    - атаки дороже mana_cap никогда не могут быть сыграны.

    0/1 knapsack по всем копиям карт с двумя ресурсами (ходы, мана).
    В оптимистической модели ВСЕ карты сразу доступны в руке, отсутствуют
    ограничения порядка добора, накопления маны и действия соперника.
    Поэтому её достижимый урон >= реального максимума в игре.
    U < starting_hp => наступление победы путём нанесения урона НЕДОСТИЖИМО.
    U >= starting_hp НЕ означает, что победа достижима.
    """
    R = rules.max_rounds
    attacks = [c for c in deck_template(rules)
               if c.effect == 'damage' and c.target == 'enemy'
               and c.value > 0 and c.cost <= rules.mana_cap]
    # dp[число_атак][потраченная_мана] = max_damage; -1 = невозможно.
    dp = [[-1] * (R + 1) for _ in range(R + 1)]
    dp[0][0] = 0
    for card in attacks:
        for plays in range(R - 1, -1, -1):
            for spent in range(R - card.cost, -1, -1):
                if dp[plays][spent] >= 0:
                    dp[plays + 1][spent + card.cost] = max(
                        dp[plays + 1][spent + card.cost],
                        dp[plays][spent] + card.value,
                    )
    U = max(map(max, dp))
    return {
        'upper_damage': U,
        'hp_required': rules.starting_hp,
        'round_budget': R,
        'mana_budget': R,
        'max_plays': R,
        'available_attack_copies': len(attacks),
        'certificate': ('0/1 knapsack, копии карт из фиксированной колоды; '
                        'общее число действий<=R, затраты маны<=R; '
                        'порядок добора и расход по раундам оптимистически игнорируются'),
    }


def strict_diagnostics(rules: GameRules):
    violations = contract_errors(rules)
    proof = optimistic_damage_bound(rules)
    if proof['upper_damage'] < rules.starting_hp:
        violations.append('PROVEN_NO_OFFENSIVE_WIN: даже оптимистическая верхняя граница '
                          f"урона U={proof['upper_damage']} < H={rules.starting_hp}")
    return violations, proof


@dataclass
class Player:
    hp: int
    mana: int = 0
    hand: list[CardRule] = field(default_factory=list)
    deck: list[CardRule] = field(default_factory=list)


class CardDuelSimulator:
    def __init__(self, rules: GameRules, seed: int):
        self.rules = rules
        self.rng = random.Random(seed)
        self.players = []
        for _ in range(2):
            deck = list(deck_template(rules))
            self.rng.shuffle(deck)
            hand = [deck.pop() for _ in range(rules.starting_hand_size)]
            self.players.append(Player(hp=rules.starting_hp, hand=hand, deck=deck))
        assert all(len(p.deck) + len(p.hand) == rules.deck_size for p in self.players)

    def run(self, policy, trace=False):
        actions = []
        for round_number in range(1, self.rules.max_rounds + 1):
            for idx in (0, 1):
                p, o = self.players[idx], self.players[1 - idx]
                p.mana = min(self.rules.mana_cap, p.mana + 1)
                if p.deck:
                    p.hand.append(p.deck.pop())
                action = policy(p, o, self.rules, self.rng)
                before_mana = p.mana
                if action is not None:
                    if action not in p.hand or action.cost > p.mana:
                        raise ValueError('агент выбрал недопустимую карту')
                    p.hand.remove(action)
                    p.mana -= action.cost
                    target = o if action.target == 'enemy' else p
                    if action.effect == 'damage':
                        target.hp -= action.value
                    elif action.effect == 'heal':
                        target.hp = min(self.rules.starting_hp, target.hp + action.value)
                    elif action.effect == 'draw' and p.deck:
                        p.hand.append(p.deck.pop())
                if trace:
                    actions.append(dict(round=round_number, player=idx, card=action.name if action else None,
                                        mana_before=before_mana, hp_p0=self.players[0].hp,
                                        hp_p1=self.players[1].hp))
                if o.hp <= 0:
                    return dict(winner=idx, rounds=round_number, reason='opponent_hp_zero', trace=actions if trace else None)
                if p.hp <= 0:
                    return dict(winner=1 - idx, rounds=round_number, reason='self_defeat', trace=actions if trace else None)
        return dict(winner=None, rounds=self.rules.max_rounds, reason='turn_limit', trace=actions if trace else None)


def random_policy(player, opponent, rules, rng):
    playable = [c for c in player.hand if c.cost <= player.mana]
    return rng.choice(playable) if playable else None


def greedy_policy(player, opponent, rules, rng):
    playable = [c for c in player.hand if c.cost <= player.mana]
    if not playable:
        return None
    attacks = [c for c in playable if c.effect == 'damage']
    if attacks:
        return max(attacks, key=lambda c: (c.value, -c.cost))
    heals = [c for c in playable if c.effect == 'heal' and player.hp <= rules.starting_hp // 2]
    if heals:
        return max(heals, key=lambda c: c.value)
    draws = [c for c in playable if c.effect == 'draw']
    return draws[0] if draws else None


def wait_for_blast_policy(player, opponent, rules, rng):
    """Иллюстративный поиск свидетельства: сберечь ману для blast."""
    return next((c for c in player.hand if c.name == 'blast' and c.effect == 'damage'
                 and c.target == 'enemy' and c.cost <= player.mana), None)


def simulate(rules, policy, n=200):
    results = [CardDuelSimulator(rules, seed).run(policy) for seed in range(n)]
    counts = Counter(x['reason'] for x in results)
    wins = Counter(x['winner'] for x in results if x['winner'] is not None)
    witness_seed = next((i for i, x in enumerate(results) if x['reason'] == 'opponent_hp_zero'), None)
    return {
        'win_rate': (n - counts['turn_limit']) / n,
        'offensive_win_rate': counts['opponent_hp_zero'] / n,
        'self_defeat_rate': counts['self_defeat'] / n,
        'turn_limit_rate': counts['turn_limit'] / n,
        'mean_rounds': round(mean(x['rounds'] for x in results), 2),
        'seat_win_gap': abs(wins[0] - wins[1]) / n,
        'witness_seed': witness_seed,
    }


def decision(violations, observed_offensive_win):
    if violations:
        return 'PROVEN_INVALID'
    if observed_offensive_win:
        return 'WITNESSED_OFFENSIVE_WIN'
    return 'INCONCLUSIVE'


def run_experiment(output_dir=Path('.')):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for name, raw, expected_valid, description in scenarios():
        rules, schema_errors = structural_validation(raw)
        if rules is None:
            raise AssertionError(f'{name} не прошёл схему: {schema_errors}')
        violations, proof = strict_diagnostics(rules)
        random_stats = simulate(rules, random_policy)
        greedy_stats = simulate(rules, greedy_policy)
        observed_greedy = greedy_stats['witness_seed'] is not None
        status = decision(violations, observed_greedy)
        # Не используем нулевую долю побед как доказательство ошибки.
        old_static = (len(original_static_diagnostics(rules)) == 0)
        row = {
            'scenario': name, 'expected_valid': expected_valid, 'description': description,
            'schema_accept': True, 'old_static_accept': old_static,
            'contract_violations': len(contract_errors(rules)),
            'upper_damage': proof['upper_damage'], 'hp_required': rules.starting_hp,
            'proven_invalid': bool(violations), 'violation_explanations': '; '.join(violations),
            'greedy_offensive_win_observed': observed_greedy,
            'evidence_status': status,
        }
        row.update({f'random_{k}': v for k, v in random_stats.items()})
        row.update({f'greedy_{k}': v for k, v in greedy_stats.items()})
        rows.append(row)
        details.append({'scenario': name, 'expected_valid': expected_valid,
                        'status': status, 'violations': violations, 'optimistic_damage_proof': proof,
                        'greedy_witness_seed': greedy_stats['witness_seed'],
                        'interpretation': ('доказано нарушение спецификации или невозможность наступательной победы'
                                           if violations else 'наблюдалась как минимум одна наступательная победа'
                                           if observed_greedy else 'отсутствие побед в выборке ничего не доказывает')})

    with (output_dir / 'validation_results_proof_aware.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'proof_certificates.json').open('w', encoding='utf-8') as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    with (output_dir / 'validation_scenarios_extended.json').open('w', encoding='utf-8') as f:
        json.dump([dict(name=name, rules=raw, expected_valid=valid, justification=descr)
                   for name, raw, valid, descr in scenarios()], f, ensure_ascii=False, indent=2)

    trap = next(rules for name, raw, _, _ in scenarios() if name == 'strategy_trap'
                for rules in [GameRules.model_validate(raw)])
    # Специальный агент не меняет статус основной оценки; лишь независимо
    # предъявляет конструктивное свидетельство достижимости для сценария.
    waiting = simulate(trap, wait_for_blast_policy, 200)
    witness = CardDuelSimulator(trap, seed=0).run(wait_for_blast_policy, trace=True)
    assert waiting['witness_seed'] is not None and witness['reason'] == 'opponent_hp_zero'
    with (output_dir / 'strategy_trap_witness.json').open('w', encoding='utf-8') as f:
        json.dump({'policy': 'wait_for_blast', 'seed': 0, 'result': witness,
                   'n_witnesses_in_200': round(waiting['offensive_win_rate'] * 200),
                   'note': 'Этот пример доказывает существование конкретной выигрышной траектории, но не универсальную играбельность'},
                  f, ensure_ascii=False, indent=2)

    print('Сценарий                  U  H  Исход  Greedy win  Доказано  Статус')
    for r in rows:
        print(f"{r['scenario']:24} {r['upper_damage']:2} {r['hp_required']:2} "
              f"{'OK' if r['expected_valid'] else 'ERR':5} {r['greedy_offensive_win_rate']:10.2f} "
              f"{str(r['proven_invalid']):8} {r['evidence_status']}")
    print('Итого:', Counter(r['evidence_status'] for r in rows))
    assert Counter(r['evidence_status'] for r in rows) == {
        'PROVEN_INVALID': 8, 'WITNESSED_OFFENSIVE_WIN': 6, 'INCONCLUSIVE': 1
    }
    assert not any(r['proven_invalid'] and r['expected_valid'] for r in rows)
    assert all(r['proven_invalid'] for r in rows if not r['expected_valid'])
    return rows


def original_static_diagnostics(rules):
    """Базовая версия из предыдущей работы — для ЧЕСТНОГО сравнения."""
    errors = contract_errors(rules)
    attacks = [c for c in rules.cards if c.effect == 'damage' and c.target == 'enemy'
               and c.value > 0 and c.cost <= rules.mana_cap]
    if not attacks:
        errors.append('нет доступных атак')
    else:
        if rules.max_rounds * max(c.value for c in attacks) < rules.starting_hp:
            errors.append('недостаточно урона по раундам')
        deck = deck_template(rules)
        if sum(c.value for c in deck if c in attacks) < rules.starting_hp:
            errors.append('недостаточно урона всей колоды')
    return errors


def test_all():
    base = GameRules.model_validate(BASE)
    sim = CardDuelSimulator(base, 0)
    assert all(len(p.deck) == 9 and len(p.hand) == 3 for p in sim.players)
    assert optimistic_damage_bound(base)['upper_damage'] >= base.starting_hp
    assert CardDuelSimulator(base, 42).run(random_policy) == CardDuelSimulator(base, 42).run(random_policy)
    corrupt = deepcopy(BASE)
    corrupt['cards'][0]['effect'] = 'teleport'
    assert structural_validation(corrupt)[0] is None

    indexed = {name: GameRules.model_validate(raw) for name, raw, _, _ in scenarios()}
    assert optimistic_damage_bound(indexed['mana_bottleneck'])['upper_damage'] == 4
    assert optimistic_damage_bound(indexed['short_horizon'])['upper_damage'] == 6
    assert optimistic_damage_bound(indexed['strategy_trap'])['upper_damage'] == 10
    assert original_static_diagnostics(indexed['mana_bottleneck']) == []
    assert strict_diagnostics(indexed['mana_bottleneck'])[0]
    assert not strict_diagnostics(indexed['strategy_trap'])[0]
    assert simulate(indexed['strategy_trap'], greedy_policy, n=200)['offensive_win_rate'] == 0
    assert simulate(indexed['strategy_trap'], wait_for_blast_policy, n=200)['offensive_win_rate'] > 0
    assert decision([], False) == 'INCONCLUSIVE'
    assert decision([], True) == 'WITNESSED_OFFENSIVE_WIN'
    assert decision(['proof'], True) == 'PROVEN_INVALID'

    # Полный перебор подмножеств короткой колоды как независимая проверка DP.
    from itertools import combinations
    for name in ('base', 'short_horizon', 'mana_bottleneck', 'strategy_trap'):
        r = indexed[name]
        attacks = [c for c in deck_template(r) if c.effect == 'damage' and c.target == 'enemy'
                   and c.value > 0 and c.cost <= r.mana_cap]
        brute = 0
        for n in range(min(len(attacks), r.max_rounds) + 1):
            for subset in combinations(attacks, n):
                if sum(c.cost for c in subset) <= r.max_rounds:
                    brute = max(brute, sum(c.value for c in subset))
        assert brute == optimistic_damage_bound(r)['upper_damage'], (name, brute)
    print('Проверки схемы, математического DP, стратегий и воспроизводимости: PASS')


if __name__ == '__main__':
    test_all()
    run_experiment(Path(__file__).parent)

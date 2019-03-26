from itertools import combinations
from collections import defaultdict


OUT_FILE = 'poker2500000.csv'
SUITS = 'CDHS'
RANKS = 'A23456789TJQK'


class Card:
    def __init__(self, i):
        self.i = i
        self.rank = (i % 13) + 1
        self.suit = (i // 13) + 1

    def as_list(self):
        return [self.rank, self.suit]

    def __str__(self):
        return RANKS[self.rank-1] + SUITS[self.suit - 1]


def check_hand(hand):
    if check_royal_flush(hand):
        return 10
    if check_straight_flush(hand):
        return 9
    if check_four_of_a_kind(hand):
        return 8
    if check_full_house(hand):
        return 7
    if check_flush(hand):
        return 6
    if check_straight(hand):
        return 5
    if check_three_of_a_kind(hand):
        return 4
    if check_two_pairs(hand):
        return 3
    if check_one_pairs(hand):
        return 2
    return 1


def check_royal_flush(hand):
    if check_flush(hand) and check_royal_straight(hand):
        return True
    return False


def check_royal_straight(hand):
    values = [i[0] for i in hand]
    if set(values) == set(["A", "K", "Q", "J", "T"]):
        return True
    return False


def check_straight_flush(hand):
    if check_flush(hand) and check_straight(hand):
        return True
    else:
        return False


def check_four_of_a_kind(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if sorted(value_counts.values()) == [1, 4]:
        return True
    return False


def check_full_house(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if sorted(value_counts.values()) == [2, 3]:
        return True
    return False


def check_flush(hand):
    suits = [i[1] for i in hand]
    if len(set(suits)) == 1:
        return True
    else:
        return False


card_order_dict = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10,
                   "J": 11, "Q": 12, "K": 13, "A": 14}


def check_straight(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    rank_values = [card_order_dict[i] for i in values]
    value_range = max(rank_values) - min(rank_values)
    if len(set(value_counts.values())) == 1 and (value_range == 4):
        return True
    else:
        # check straight with low Ace
        if set(values) == set(["A", "2", "3", "4", "5"]):
            return True
        return False


def check_three_of_a_kind(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if set(value_counts.values()) == set([3, 1]):
        return True
    else:
        return False


def check_two_pairs(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if sorted(value_counts.values()) == [1, 2, 2]:
        return True
    else:
        return False


def check_one_pairs(hand):
    values = [i[0] for i in hand]
    value_counts = defaultdict(lambda: 0)
    for v in values:
        value_counts[v] += 1
    if 2 in value_counts.values():
        return True
    else:
        return False


cards_i = list(range(52))

total = 2598960

if __name__ == '__main__':
    with open(OUT_FILE, 'w') as out_file:
        i = 0
        for hand in combinations(cards_i, 5):
            cards = [Card(i) for i in hand]
            hand_class = check_hand(list(map(lambda c: str(c), cards)))

            flattened = []
            for card in cards:
                flattened += card.as_list()
            flattened += [hand_class]

            line = ','.join(map(str, flattened))
            # print(line)
            out_file.write(f'{line}\n')
            i += 1
            print(f'{"%.1f" % (i * 100 / total)}%', end='\r')

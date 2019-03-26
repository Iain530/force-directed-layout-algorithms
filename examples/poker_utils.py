import numpy as np


def load_file(name, dtype):
    with open(f'../datasets/{name}', encoding='utf8') as data_file:
        return np.loadtxt(
            data_file,
            skiprows=1,
            delimiter=',',
            dtype=dtype,
            comments='#'
        )


def load_poker(size):
    return load_file(f'poker/poker{size}.csv', np.int16)


# suits = ['♣', '♦', '♥', '♠']
# ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', ]
hand_classes = [
    'High Card',
    'Pair',
    'Two pair',
    'Three of a kind',
    'Straight',
    'Flush',
    'Full house',
    'Four of a kind',
    'Straight flush',
    'Royal flush',
]
cards = ['🃑', '🃒', '🃓', '🃔', '🃕', '🃖', '🃗', '🃘', '🃙', '🃚', '🃛', '🃝', '🃞',
         '🃁', '🃂', '🃃', '🃄', '🃅', '🃆', '🃇', '🃈', '🃉', '🃊', '🃋', '🃍', '🃎',
         '🂱', '🂲', '🂳', '🂴', '🂵', '🂶', '🂷', '🂸', '🂹', '🂺', '🂻', '🂽', '🂾',
         '🂡', '🂢', '🂣', '🂤', '🂥', '🂦', '🂧', '🂨', '🂩', '🂪', '🂫', '🂭', '🂮']


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)


def annotate_poker(node, i):
    def card(suit, rank):
        return cards[((suit-1)*13) + rank - 1]

    arr = node.datapoint
    hand = []
    for s, r in grouped(arr[:10], 2):
        hand.append(card(s, r))
    if (len(arr) > 10):
        hand.append(hand_classes[arr[10]])
    return ' '.join(hand)


def poker_distance(h1, h2):
    """
    similarity metric between two poker hands
    """
    ranks = range(0, 9, 2)
    suits = range(1, 10, 2)

    h1_ranks = h1[ranks]
    h2_ranks = h2[ranks]
    h1_ranks.sort()
    h2_ranks.sort()

    rank_diff = abs(sum(h2_ranks - h1_ranks)) / 75

    h1_suits = h1[suits]
    h2_suits = h2[suits]
    h1_suits.sort()
    h2_suits.sort()

    suit_diff = 1
    i = j = 0
    while i < 5 and j < 5:
        comp = h2_suits[j] - h1_suits[i]
        if comp == 0:
            suit_diff -= 0.2
            i += 1
            j += 1
        elif comp < 0:
            j += 1
        else:
            i += 1

    class_diff = abs(h1[10] - h2[10])
    return rank_diff + suit_diff + class_diff

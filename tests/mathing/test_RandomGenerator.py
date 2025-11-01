
import random
import pytest

# Adjust this import to wherever RandomGenerator lives.
from faxai.mathing.RandomGenerator import RandomGenerator


@pytest.mark.parametrize("seed", [0, 1, 42, 9999])
def test_reproducible_rand(seed):
    a = RandomGenerator(seed)
    b = RandomGenerator(seed)
    seq_a = [a.rand() for _ in range(5)]
    seq_b = [b.rand() for _ in range(5)]
    assert seq_a == seq_b
    # Values are in [0.0, 1.0)
    assert all(0.0 <= x < 1.0 for x in seq_a)


@pytest.mark.parametrize("seed", [0, 13, 123456])
def test_random_alias(seed):
    rng = RandomGenerator(seed)
    # The alias points to the same function object
    assert rng.random.__func__ is rng.rand.__func__
    # Both produce valid floats
    vals = [rng.random() for _ in range(3)]
    assert all(isinstance(v, float) and 0.0 <= v < 1.0 for v in vals)


@pytest.mark.parametrize("seed", [0, 17])
def test_randint_half_open_interval_and_repro(seed):
    low, high = 2, 9  # [2, 9)
    a = RandomGenerator(seed)
    b = RandomGenerator(seed)
    seq_a = [a.randint(low, high) for _ in range(50)]
    seq_b = [b.randint(low, high) for _ in range(50)]
    assert seq_a == seq_b
    assert all(low <= v < high for v in seq_a)


def test_randint_invalid_range_raises():
    rng = RandomGenerator(0)
    with pytest.raises(ValueError):
        rng.randint(5, 5)
    with pytest.raises(ValueError):
        rng.randint(10, 5)


@pytest.mark.parametrize("seed", [0, 5, 1234])
def test_choice_reproducible(seed):
    seq = ["a", "b", "c", "d"]
    a = RandomGenerator(seed)
    b = RandomGenerator(seed)
    a_choices = [a.choice(seq) for _ in range(10)]
    b_choices = [b.choice(seq) for _ in range(10)]
    assert a_choices == b_choices
    assert all(ch in seq for ch in a_choices)


def test_choice_empty_raises():
    rng = RandomGenerator(0)
    with pytest.raises(IndexError):
        rng.choice([])


@pytest.mark.parametrize("seed", [0, 99])
def test_shuffle_in_place_and_reproducible(seed):
    base = list(range(10))
    x = base[:]  # for first shuffle
    y = base[:]  # for second RNG, same seed

    a = RandomGenerator(seed)
    b = RandomGenerator(seed)

    ret_a = a.shuffle(x)
    ret_b = b.shuffle(y)

    # Returns None, and shuffles in place
    assert ret_a is None and ret_b is None
    assert sorted(x) == sorted(base) == sorted(y)

    # Reproducible across instances with same seed
    assert x == y

    # Resetting seed reproduces the same shuffle result again
    a.reset_seed()
    z = base[:]
    a.shuffle(z)
    assert z == x


def test_set_seed_switches_stream_reproducibly():
    # Stream S0 from seed=0, then switch to seed=123
    r = RandomGenerator(0)
    s0_first = [r.rand() for _ in range(3)]
    r.set_seed(123)
    s123_from_r = [r.rand() for _ in range(3)]

    # Fresh instance for seed=123 should match the second segment
    r123 = RandomGenerator(123)
    s123_fresh = [r123.rand() for _ in range(3)]
    assert s123_from_r == s123_fresh

    # Fresh instance for seed=0 should match the first segment (if started fresh)
    r0 = RandomGenerator(0)
    assert s0_first == [r0.rand() for _ in range(3)]


def test_reset_seed_returns_to_initial_stream():
    r = RandomGenerator(42)
    seq1 = [r.rand() for _ in range(5)]
    # Advance RNG arbitrarily
    _ = [r.randint(1, 100) for _ in range(7)]
    r.reset_seed()
    seq2 = [r.rand() for _ in range(5)]
    assert seq1 == seq2


def test_seed_none_is_respected_and_resettable(monkeypatch):
    # Force the internally-chosen seed to be a known value
    monkeypatch.setattr(random, "randint", lambda a, b: 12345)
    r = RandomGenerator(seed=None)
    # Should act exactly like an instance constructed with seed=12345
    r_known = RandomGenerator(12345)

    seq_r = [r.rand() for _ in range(5)]
    seq_known = [r_known.rand() for _ in range(5)]
    assert seq_r == seq_known

    # Reset should also reproduce the same stream from the beginning
    r.reset_seed()
    assert [r.rand() for _ in range(5)] == seq_known


@pytest.mark.parametrize("seed", [0, 7])
def test_methods_do_not_leak_state_between_instances(seed):
    # Ensure two instances remain independent
    a = RandomGenerator(seed)
    b = RandomGenerator(seed)

    _ = a.rand()
    _ = b.randint(1, 10)
    _ = a.choice([1, 2, 3])
    _ = b.choice(["x", "y"])

    # After different calls, reseeding both to the same seed brings them back in sync
    a.set_seed(seed)
    b.set_seed(seed)
    seq_a = [a.rand(), a.randint(3, 9), a.choice([10, 20, 30])]
    seq_b = [b.rand(), b.randint(3, 9), b.choice([10, 20, 30])]
    assert seq_a == seq_b

def test_none_seed_produces_different_instances():
    """
    Test that two RandomGenerator instances created with seed=None
    usually produce different random streams.
    Because collisions are theoretically possible, we allow up to one accidental match.
    """
    TRIALS = 64
    STREAM_LEN = 5

    clashes = 0
    for _ in range(TRIALS):
        a = RandomGenerator(seed=None)
        b = RandomGenerator(seed=None)
        seq_a = [a.rand() for _ in range(STREAM_LEN)]
        seq_b = [b.rand() for _ in range(STREAM_LEN)]
        if seq_a == seq_b:
            clashes += 1
            if clashes > 1:
                break

    assert clashes <= 1, f"Found {clashes} identical streams out of {TRIALS} trials; expected at most 1."

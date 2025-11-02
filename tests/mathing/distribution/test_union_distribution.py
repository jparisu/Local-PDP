import numpy as np
import pytest

from faxai.mathing.distribution.parametric_distributions import NormalDistribution, UniformDistribution
from faxai.mathing.distribution.UnionDistribution import UnionDistribution
from faxai.mathing.RandomGenerator import RandomGenerator

# --- Helpers -----------------------------------------------------------------

def mixture_mean(means):
    """Equal-weight mixture mean."""
    return float(np.mean(means))

def mixture_var(component_vars, component_means):
    """Equal-weight mixture variance: E[Var] + Var(E)."""
    return float(np.mean(component_vars) + np.var(component_means, ddof=0))

# --- Tests: Normal + Uniform --------------------------------------------------

def test_normal_uniform_moments():
    # Components
    mu, sigma = 2.5, 1.2
    a, b = -1.0, 3.0

    normal = NormalDistribution(mu, sigma)
    uniform = UniformDistribution(a, b)
    mix = UnionDistribution([normal, uniform])

    # Analytical targets
    m_u = 0.5 * (a + b)
    var_u = (b - a) ** 2 / 12.0

    target_mean = mixture_mean([mu, m_u])
    target_var  = mixture_var([sigma**2, var_u], [mu, m_u])

    assert mix.mean() == pytest.approx(target_mean, rel=0, abs=1e-12)
    assert mix.std()**2 == pytest.approx(target_var, rel=1e-12, abs=0)

def test_normal_uniform_pdf_and_cdf_consistency():
    mu, sigma = 0.0, 2.0
    a, b = -3.0, 5.0

    normal = NormalDistribution(mu, sigma)
    uniform = UniformDistribution(a, b)
    mix = UnionDistribution([normal, uniform])

    # PDF should be average of component PDFs
    xs = np.linspace(-10, 10, 9)
    for x in xs:
        expected_pdf = 0.5 * (normal.pdf(x) + uniform.pdf(x))
        assert mix.pdf(x) == pytest.approx(expected_pdf, rel=1e-12, abs=0)

    # CDF should be average of component CDFs (both have exact cdf)
    for x in xs:
        expected_cdf = 0.5 * (normal.cdf(x) + uniform.cdf(x))
        assert mix.cdf(x) == pytest.approx(expected_cdf, rel=1e-12, abs=0)

    # CDF edge behavior
    assert mix.cdf(-1e6) == pytest.approx(0.0, abs=1e-12)
    assert mix.cdf( 1e6) == pytest.approx(1.0, abs=1e-12)


def test_normal_uniform_empirical_moments():
    mu, sigma = 1.0, 0.7
    a, b = -2.0, 4.0

    normal = NormalDistribution(mu, sigma)
    uniform = UniformDistribution(a, b)
    mix = UnionDistribution([normal, uniform])

    # Targets
    m_u = 0.5 * (a + b)
    var_u = (b - a) ** 2 / 12.0
    target_mean = mixture_mean([mu, m_u])
    target_std  = np.sqrt(mixture_var([sigma**2, var_u], [mu, m_u]))

    rng = RandomGenerator(2025)
    n = 50_000
    samples = mix.random_sample(n=n, rng=rng)
    emp_mean = float(np.mean(samples))
    emp_std  = float(np.std(samples, ddof=0))

    # Tolerances scale with std / sqrt(n)
    mean_tol = 4 * target_std / np.sqrt(n)   # ~4 sigma tolerance for sample mean
    std_tol  = 0.03 * target_std + 1e-3      # allow ~3% relative slack

    assert emp_mean == pytest.approx(target_mean, abs=mean_tol)
    assert emp_std  == pytest.approx(target_std,  rel=0.03, abs=std_tol)

# --- Tests: Three Normals -----------------------------------------------------

def test_three_normals_moments():
    comps = [
        NormalDistribution(-1.0, 0.5),
        NormalDistribution( 0.5, 1.0),
        NormalDistribution( 3.0, 2.0),
    ]
    mix = UnionDistribution(comps)

    means = np.array([c.mean() for c in comps], dtype=float)
    vars_ = np.array([c.std()**2 for c in comps], dtype=float)

    target_mean = mixture_mean(means)
    target_var  = mixture_var(vars_, means)

    assert mix.mean() == pytest.approx(target_mean, abs=1e-12)
    assert mix.std()**2 == pytest.approx(target_var,  rel=1e-12, abs=0)

def test_three_normals_pdf_cdf_and_empirical():
    comps = [
        NormalDistribution(-2.0, 0.8),
        NormalDistribution( 0.0, 1.2),
        NormalDistribution( 3.0, 0.6),
    ]
    mix = UnionDistribution(comps)

    # PDF/CDF are averages
    xs = np.linspace(-6, 6, 13)
    for x in xs:
        expected_pdf = np.mean([c.pdf(x) for c in comps])
        expected_cdf = np.mean([c.cdf(x) for c in comps])
        assert mix.pdf(x) == pytest.approx(expected_pdf, rel=1e-12, abs=0)
        assert mix.cdf(x) == pytest.approx(expected_cdf, rel=1e-12, abs=0)

    # Empirical sanity check
    rng = RandomGenerator(7)
    n = 50_000
    samples = mix.random_sample(n=n, rng=rng)

    target_mean = mixture_mean([c.mean() for c in comps])
    target_std  = np.sqrt(mixture_var([c.std()**2 for c in comps],
                                      [c.mean() for c in comps]))

    emp_mean = float(np.mean(samples))
    emp_std  = float(np.std(samples, ddof=0))

    mean_tol = 4 * target_std / np.sqrt(n)
    assert emp_mean == pytest.approx(target_mean, abs=mean_tol)
    assert emp_std  == pytest.approx(target_std,  rel=0.03, abs=0.003)

    # Max CDF still 1; PDF nonnegative
    assert all(mix.pdf(x) >= 0.0 for x in xs)

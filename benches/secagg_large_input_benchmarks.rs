#![allow(non_snake_case)]

extern crate rand;
use rand::rngs::OsRng;
use rand::thread_rng;

#[macro_use]
extern crate criterion;

use criterion::BatchSize;
use criterion::Criterion;

extern crate curve25519_dalek_ng;
use curve25519_dalek_ng as curve25519_dalek;

use curve25519_dalek::constants;
use curve25519_dalek::scalar::Scalar;

static BATCH_SIZES: [usize; 5] = [1, 2, 4, 8, 16];
static MULTISCALAR_SIZES: [usize; 14] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

static SECAGG_FULL_COMMIT_SIZES: [usize; 0] = [];
static SECAGG_TRI_COMMIT_SIZES: [usize; 0] = [];
static SECAGG_VAR_COMMIT_SIZES: [(usize, u64); 19] = [
    (128, 83886080000), // for committing to y
    (32, 65536), (64, 65536), (128, 65536), (256, 65536), (512, 65536), (1024, 65536), (2048, 65536), (4096, 65536), (8192, 65536), (16384, 65536), (32768, 65536), (65536, 65536), (131072, 65536), (262144, 65536), (524288, 65536), (1048576, 65536), (2097152, 65536), (4194304, 65536), // for committing to z
];

mod edwards_benches {
    use super::*;

    use curve25519_dalek::edwards::EdwardsPoint;

    fn compress(c: &mut Criterion) {
        let B = &constants::ED25519_BASEPOINT_POINT;
        c.bench_function("EdwardsPoint compression", move |b| b.iter(|| B.compress()));
    }

    fn decompress(c: &mut Criterion) {
        let B_comp = &constants::ED25519_BASEPOINT_COMPRESSED;
        c.bench_function("EdwardsPoint decompression", move |b| {
            b.iter(|| B_comp.decompress().unwrap())
        });
    }

    fn consttime_fixed_base_scalar_mul(c: &mut Criterion) {
        let B = &constants::ED25519_BASEPOINT_TABLE;
        let s = Scalar::from(897987897u64).invert();
        c.bench_function("Constant-time fixed-base scalar mul", move |b| {
            b.iter(|| B * &s)
        });
    }

    fn consttime_variable_base_scalar_mul(c: &mut Criterion) {
        let B = &constants::ED25519_BASEPOINT_POINT;
        let s = Scalar::from(897987897u64).invert();
        c.bench_function("Constant-time variable-base scalar mul", move |b| {
            b.iter(|| B * s)
        });
    }

    fn vartime_double_base_scalar_mul(c: &mut Criterion) {
        c.bench_function("Variable-time aA+bB, A variable, B fixed", |bench| {
            let mut rng = thread_rng();
            let A = &Scalar::random(&mut rng) * &constants::ED25519_BASEPOINT_TABLE;
            bench.iter_batched(
                || (Scalar::random(&mut rng), Scalar::random(&mut rng)),
                |(a, b)| EdwardsPoint::vartime_double_scalar_mul_basepoint(&a, &A, &b),
                BatchSize::SmallInput,
            );
        });
    }

    criterion_group! {
        name = edwards_benches;
        config = Criterion::default();
        targets =
        compress,
        decompress,
        consttime_fixed_base_scalar_mul,
        consttime_variable_base_scalar_mul,
        vartime_double_base_scalar_mul,
    }
}

mod multiscalar_benches {
    use super::*;

    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::edwards::VartimeEdwardsPrecomputation;
    use curve25519_dalek::traits::MultiscalarMul;
    use curve25519_dalek::traits::VartimeMultiscalarMul;
    use curve25519_dalek::traits::VartimePrecomputedMultiscalarMul;
    use curve25519_dalek_ng::traits::Identity;
    use rand::prelude::SliceRandom;
    use rand::Rng;

    fn construct_scalars(n: usize) -> Vec<Scalar> {
        let mut rng = thread_rng();
        (0..n).map(|_| Scalar::random(&mut rng)).collect()
    }

    // Constructs scalars in {-1, 0, 1} with even distribution
    fn construct_small_restricted_scalars(n: usize) -> Vec<Scalar> {
        let mut rng = thread_rng();
        let restricted_scalars = vec![-Scalar::one(), Scalar::zero(), Scalar::one()];
        let output: Vec<Scalar> = (0..n).map(|_| *restricted_scalars.choose(&mut rng).unwrap())
            .collect();
        output
    }

    // Constructs scalars in [-bound/2, bound/2) with even distribution
    fn construct_variable_restricted_scalars(n: usize, bound: u64) -> Vec<Scalar> {
        let mut rng = thread_rng();

        (0..n).map(|_| {
            // Can't generate a scalar from a negative number, so generate positive scalars and negate half of them.
            let mut rand_scalar = Scalar::from(rng.gen_range(0..bound/2));
            if rng.gen_range(0..2) == 1 {
                rand_scalar = -rand_scalar;
            }
            rand_scalar
        }).collect()
    }

    // Return ints in {-1, 0, 1} with even distribution
    fn construct_restricted_ints(n: usize) -> Vec<i32> {
        let mut rng = thread_rng();
        let restricted_ints = vec![-1, 0, 1];
        let output: Vec<i32> = (0..n).map(|_| *restricted_ints.choose(&mut rng).unwrap())
            .collect();
        output
    }

    fn construct_points(n: usize) -> Vec<EdwardsPoint> {
        let mut rng = thread_rng();
        (0..n)
            .map(|_| &Scalar::random(&mut rng) * &constants::ED25519_BASEPOINT_TABLE)
            .collect()
    }

    fn construct(n: usize) -> (Vec<Scalar>, Vec<EdwardsPoint>) {
        (construct_scalars(n), construct_points(n))
    }

    fn consttime_multiscalar_mul(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Constant-time variable-base multiscalar multiplication",
            |b, &&size| {
                let points = construct_points(size);
                // This is supposed to be constant-time, but we might as well
                // rerandomize the scalars for every call just in case.
                b.iter_batched(
                    || construct_scalars(size),
                    |scalars| EdwardsPoint::multiscalar_mul(&scalars, &points),
                    BatchSize::SmallInput,
                );
            },
            &MULTISCALAR_SIZES,
        );
    }

    fn vartime_multiscalar_mul(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "SecAgg: scalars in full range",
            |b, &&size| {
                let points = construct_points(size);
                // Rerandomize the scalars for every call to prevent
                // false timings from better caching (e.g., the CPU
                // cache lifts exactly the right table entries for the
                // benchmark into the highest cache levels).
                b.iter_batched(
                    || construct_scalars(size),
                    |scalars| EdwardsPoint::vartime_multiscalar_mul(&scalars, &points),
                    BatchSize::SmallInput,
                );
            },
            &SECAGG_FULL_COMMIT_SIZES,
        );
    }

    // Multiscalar mul with scalars in [-bound/2, bound/2) with even distribution
    fn variable_restricted_vartime_multiscalar_mul(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "SecAgg: scalars in [-bound/2, bound/2)",
            |b, &&params| {
                let (size, bound) = params;
                let points = construct_points(size);
                // Rerandomize the scalars for every call to prevent
                // false timings from better caching (e.g., the CPU
                // cache lifts exactly the right table entries for the
                // benchmark into the highest cache levels).
                b.iter_batched(
                    || construct_variable_restricted_scalars(size, bound),
                    |scalars| EdwardsPoint::vartime_multiscalar_mul(&scalars, &points),
                    BatchSize::SmallInput,
                );
            },
            &SECAGG_VAR_COMMIT_SIZES,
        );
    }

    // This is the sum from using multiscalar mults
    fn restricted_scalars_mul_fast(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Variable-time restricted-scalar-mul",
            |b, &&size| {
                let points = construct_points(size);
                // Rerandomize the scalars for every call to prevent
                // false timings from better caching (e.g., the CPU
                // cache lifts exactly the right table entries for the
                // benchmark into the highest cache levels).
                b.iter_batched(
                    || construct_small_restricted_scalars(size),
                    |scalars| {
                        assert_eq!(points.len(), scalars.len());
                        // println!("points: {:?}", points);
                        // println!("scalars: {:?}", scalars);
                        EdwardsPoint::vartime_multiscalar_mul(&scalars, &points);
                    },
                    BatchSize::SmallInput,
                );
            },
            &MULTISCALAR_SIZES,
        );
    }

    // This is the sum from using naive multiplication and addition
    fn restricted_scalars_mul_naive(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Variable-time restricted-scalar-mul using naive multiplication",
            |b, &&size| {
                let points = construct_points(size);
                // Rerandomize the scalars for every call to prevent
                // false timings from better caching (e.g., the CPU
                // cache lifts exactly the right table entries for the
                // benchmark into the highest cache levels).
                b.iter_batched(
                    || construct_restricted_ints(size),
                    |ints| {
                        let _mult_sum: EdwardsPoint = points
                            .iter()
                            .zip(ints.iter())
                            .map(|(point, int)| point * match int {
                                1 => Scalar::one(),
                                0 => Scalar::zero(),
                                -1 => -Scalar::one(),
                                _ => Scalar::zero(),
                            })
                            .sum();
                    },
                    BatchSize::SmallInput,
                );
            },
            &MULTISCALAR_SIZES,
        );
    }

    // This is the sum from using simple selection over {-1, 0, 1}
    fn restricted_scalars_add(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "SecAgg: scalars in {-1, 0, 1}",
            |b, &&size| {
                let points = construct_points(size);
                // Rerandomize the scalars for every call to prevent
                // false timings from better caching (e.g., the CPU
                // cache lifts exactly the right table entries for the
                // benchmark into the highest cache levels).
                b.iter_batched(
                    || construct_restricted_ints(size),
                    |ints| {
                        let _: EdwardsPoint = points
                            .iter()
                            .zip(ints.iter())
                            .map(|(point, int)| match int {
                                1 => *point,
                                0 => EdwardsPoint::identity(),
                                -1 => -point,
                                _ => EdwardsPoint::identity(),
                            })
                            .sum();
                    },
                    BatchSize::SmallInput,
                );
            },
            &SECAGG_TRI_COMMIT_SIZES,
        );
    }

    fn vartime_precomputed_pure_static(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Variable-time fixed-base multiscalar multiplication",
            move |b, &&total_size| {
                let static_size = total_size;

                let static_points = construct_points(static_size);
                let precomp = VartimeEdwardsPrecomputation::new(&static_points);
                // Rerandomize the scalars for every call to prevent
                // false timings from better caching (e.g., the CPU
                // cache lifts exactly the right table entries for the
                // benchmark into the highest cache levels).
                b.iter_batched(
                    || construct_scalars(static_size),
                    |scalars| precomp.vartime_multiscalar_mul(&scalars),
                    BatchSize::SmallInput,
                );
            },
            &MULTISCALAR_SIZES,
        );
    }

    fn vartime_precomputed_helper(c: &mut Criterion, dynamic_fraction: f64) {
        let label = format!(
            "Variable-time mixed-base multiscalar multiplication ({:.0}pct dyn)",
            100.0 * dynamic_fraction,
        );
        c.bench_function_over_inputs(
            &label,
            move |b, &&total_size| {
                let dynamic_size = ((total_size as f64) * dynamic_fraction) as usize;
                let static_size = total_size - dynamic_size;

                let static_points = construct_points(static_size);
                let dynamic_points = construct_points(dynamic_size);
                let precomp = VartimeEdwardsPrecomputation::new(&static_points);
                // Rerandomize the scalars for every call to prevent
                // false timings from better caching (e.g., the CPU
                // cache lifts exactly the right table entries for the
                // benchmark into the highest cache levels).  Timings
                // should be independent of points so we don't
                // randomize them.
                b.iter_batched(
                    || {
                        (
                            construct_scalars(static_size),
                            construct_scalars(dynamic_size),
                        )
                    },
                    |(static_scalars, dynamic_scalars)| {
                        precomp.vartime_mixed_multiscalar_mul(
                            &static_scalars,
                            &dynamic_scalars,
                            &dynamic_points,
                        )
                    },
                    BatchSize::SmallInput,
                );
            },
            &MULTISCALAR_SIZES,
        );
    }

    fn vartime_precomputed_00_pct_dynamic(c: &mut Criterion) {
        vartime_precomputed_helper(c, 0.0);
    }

    fn vartime_precomputed_20_pct_dynamic(c: &mut Criterion) {
        vartime_precomputed_helper(c, 0.2);
    }

    fn vartime_precomputed_50_pct_dynamic(c: &mut Criterion) {
        vartime_precomputed_helper(c, 0.5);
    }

    criterion_group! {
        name = multiscalar_benches;
        // Lower the sample size to run the benchmarks faster
        config = Criterion::default().sample_size(10);
        targets =
        // consttime_multiscalar_mul,
        // vartime_precomputed_pure_static,
        // vartime_precomputed_00_pct_dynamic,
        // vartime_precomputed_20_pct_dynamic,
        // vartime_precomputed_50_pct_dynamic,
        // restricted_scalars_mul_fast,
        // restricted_scalars_mul_naive,
        // vartime_multiscalar_mul, // Multiscalar mul over the entire range of scalars
        // restricted_scalars_add,  // Multiscalar mul over scalars in {-1, 0, 1} (aka no multiplication, just point addition)
        variable_restricted_vartime_multiscalar_mul, // Multiscalar mul over scalars in a variable range
    }
}

mod ristretto_benches {
    use super::*;
    use curve25519_dalek::ristretto::RistrettoPoint;

    fn compress(c: &mut Criterion) {
        c.bench_function("RistrettoPoint compression", |b| {
            let B = &constants::RISTRETTO_BASEPOINT_POINT;
            b.iter(|| B.compress())
        });
    }

    fn decompress(c: &mut Criterion) {
        c.bench_function("RistrettoPoint decompression", |b| {
            let B_comp = &constants::RISTRETTO_BASEPOINT_COMPRESSED;
            b.iter(|| B_comp.decompress().unwrap())
        });
    }

    fn double_and_compress_batch(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Batch Ristretto double-and-encode",
            |b, &&size| {
                let mut rng = OsRng;
                let points: Vec<RistrettoPoint> = (0..size)
                    .map(|_| RistrettoPoint::random(&mut rng))
                    .collect();
                b.iter(|| RistrettoPoint::double_and_compress_batch(&points));
            },
            &BATCH_SIZES,
        );
    }

    criterion_group! {
        name = ristretto_benches;
        config = Criterion::default();
        targets =
        compress,
        decompress,
        double_and_compress_batch,
    }
}

mod montgomery_benches {
    use super::*;

    fn montgomery_ladder(c: &mut Criterion) {
        c.bench_function("Montgomery pseudomultiplication", |b| {
            let B = constants::X25519_BASEPOINT;
            let s = Scalar::from(897987897u64).invert();
            b.iter(|| B * s);
        });
    }

    criterion_group! {
        name = montgomery_benches;
        config = Criterion::default();
        targets = montgomery_ladder,
    }
}

mod scalar_benches {
    use super::*;

    fn scalar_inversion(c: &mut Criterion) {
        c.bench_function("Scalar inversion", |b| {
            let s = Scalar::from(897987897u64).invert();
            b.iter(|| s.invert());
        });
    }

    fn batch_scalar_inversion(c: &mut Criterion) {
        c.bench_function_over_inputs(
            "Batch scalar inversion",
            |b, &&size| {
                let mut rng = OsRng;
                let scalars: Vec<Scalar> = (0..size).map(|_| Scalar::random(&mut rng)).collect();
                b.iter(|| {
                    let mut s = scalars.clone();
                    Scalar::batch_invert(&mut s);
                });
            },
            &BATCH_SIZES,
        );
    }

    criterion_group! {
        name = scalar_benches;
        config = Criterion::default();
        targets =
        scalar_inversion,
        batch_scalar_inversion,
    }
}

criterion_main!(
    scalar_benches::scalar_benches,
    montgomery_benches::montgomery_benches,
    ristretto_benches::ristretto_benches,
    edwards_benches::edwards_benches,
    multiscalar_benches::multiscalar_benches,
);

//! Numeric statistics helpers — sample variance/stdev, mean, median.

pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum: f64 = values.iter().sum();
    let m = sum / values.len() as f64;
    if m.is_finite() { Some(m) } else { None }
}

pub fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m = mean(values)?;
    let sum_sq_diff: f64 = values.iter().map(|v| (v - m).powi(2)).sum();
    let var = sum_sq_diff / (values.len() - 1) as f64;
    if var.is_finite() { Some(var) } else { None }
}

pub fn sample_stdev(values: &[f64]) -> Option<f64> {
    sample_variance(values).map(f64::sqrt)
}

pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n.is_multiple_of(2) {
        let lo = sorted[n / 2 - 1];
        let hi = sorted[n / 2];
        Some((lo + hi) / 2.0)
    } else {
        Some(sorted[n / 2])
    }
}

pub fn format_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let abs = x.abs();
    let exp = abs.log10().floor() as i32;
    if !(-4..4).contains(&exp) {
        let s = format!("{:.3e}", x);
        normalize_scientific_exp(&s)
    } else {
        let digits_after = (3 - exp).max(0) as usize;
        let s = format!("{:.*}", digits_after, x);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

fn normalize_scientific_exp(s: &str) -> String {
    let Some(epos) = s.find('e') else {
        return s.to_string();
    };
    let (mantissa, rest) = s.split_at(epos);
    let exp_part = &rest[1..];
    let exp_num: i32 = exp_part.parse().unwrap_or(0);
    let mantissa_clean = if mantissa.contains('.') {
        mantissa
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    } else {
        mantissa.to_string()
    };
    let sign = if exp_num >= 0 { "+" } else { "-" };
    format!("{}e{}{:02}", mantissa_clean, sign, exp_num.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_basic() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn sample_variance_n_minus_1() {
        let v = sample_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((v - 2.5).abs() < 1e-9);
    }

    #[test]
    fn median_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn median_even() {
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
    }

    #[test]
    fn format_g_basic() {
        assert_eq!(format_g(1.5), "1.5");
        assert_eq!(format_g(1.0), "1");
        assert_eq!(format_g(12345.678), "1.235e+04");
        assert_eq!(format_g(0.00001234), "1.234e-05");
    }
}

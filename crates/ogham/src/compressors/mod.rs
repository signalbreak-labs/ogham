pub mod ast_code;
pub mod dedup_ref;
pub mod focus;
pub mod log_stripper;
pub mod semantic;
pub mod smart_crusher;
pub mod toon;

/// Largest prefix of `s` no longer than `max` bytes that ends on a UTF-8 char
/// boundary, so truncating never panics on multibyte content.
pub(crate) fn truncate_on_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

//! Provider-native context-management adapters.
//!
//! Some providers now offer server-side context editing. These adapters
//! translate Ogham policies into the provider's wire format so a host can
//! delegate the basics to the platform while keeping Ogham as the
//! provider-agnostic layer (and the only reversible one). Adapters are pure
//! data-structure builders — Ogham never talks to providers itself.

pub mod anthropic;
pub mod gemini;
pub mod openai;

use ogham_core::Message;

/// Deterministic content identity for a span of messages.
///
/// Provider cache planners use this to key provider-side caches and to detect
/// when cached content must be refreshed: identical spans yield identical keys.
pub fn content_key(messages: &[Message]) -> String {
    let joined = messages
        .iter()
        .map(|m| format!("{}\n{}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n---\n");
    format!("ogham-{}", crate::ccr::compute_key(joined.as_bytes()))
}

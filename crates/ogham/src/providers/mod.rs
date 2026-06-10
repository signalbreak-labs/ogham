//! Provider-native context-management adapters.
//!
//! Some providers now offer server-side context editing. These adapters
//! translate Ogham policies into the provider's wire format so a host can
//! delegate the basics to the platform while keeping Ogham as the
//! provider-agnostic layer (and the only reversible one). Adapters are pure
//! data-structure builders — Ogham never talks to providers itself.

pub mod anthropic;

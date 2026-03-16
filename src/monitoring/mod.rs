use anyhow::Result;

/// Notification & monitoring: Telegram, Discord, logging.
pub struct Notifier {
    telegram_token: Option<String>,
    telegram_chat_id: Option<String>,
    discord_webhook: Option<String>,
}

impl Notifier {
    pub fn new(
        telegram_token: Option<String>,
        telegram_chat_id: Option<String>,
        discord_webhook: Option<String>,
    ) -> Self {
        Self {
            telegram_token,
            telegram_chat_id,
            discord_webhook,
        }
    }

    pub async fn send(&self, message: &str) -> Result<()> {
        tracing::info!(msg = message, "Sending notification");

        if let (Some(_token), Some(_chat_id)) = (&self.telegram_token, &self.telegram_chat_id) {
            // TODO: send via Telegram Bot API
        }

        if let Some(_webhook) = &self.discord_webhook {
            // TODO: send via Discord webhook
        }

        Ok(())
    }
}

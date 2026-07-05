import { Eye, EyeOff, Save } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { fetchSettings, saveSettings } from "../api";
import type { SettingsForm } from "../types";

const emptySettings: SettingsForm = {
  base_url: "",
  api_key: "",
  model: ""
};

export function SettingsPanel() {
  const [settings, setSettings] = useState<SettingsForm>(emptySettings);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    fetchSettings()
      .then((data) => {
        if (alive) setSettings(data);
      })
      .catch((err) => {
        if (alive) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setMessage("");
    setError("");
    try {
      const saved = await saveSettings(settings);
      setSettings(saved);
      setMessage("设置已保存，后续问答会使用新的模型配置。");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="settings-pane">
      <header className="settings-header">
        <div>
          <h1>系统设置</h1>
          <p>配置 OpenAI 兼容接口，保存后会写入项目根目录 .env。</p>
        </div>
      </header>

      <form className="settings-form" onSubmit={handleSubmit}>
        <label className="setting-field">
          <span>Base URL</span>
          <input
            value={settings.base_url}
            onChange={(event) => setSettings((current) => ({ ...current, base_url: event.target.value }))}
            placeholder="https://your-openai-compatible-endpoint/v1"
            disabled={loading}
          />
        </label>

        <label className="setting-field">
          <span>API Key</span>
          <div className="secret-input">
            <input
              value={settings.api_key}
              onChange={(event) => setSettings((current) => ({ ...current, api_key: event.target.value }))}
              type={showKey ? "text" : "password"}
              placeholder="sk-..."
              autoComplete="current-password"
              disabled={loading}
            />
            <button type="button" onClick={() => setShowKey((value) => !value)} title={showKey ? "隐藏" : "显示"}>
              {showKey ? <EyeOff size={17} /> : <Eye size={17} />}
            </button>
          </div>
        </label>

        <label className="setting-field">
          <span>Model</span>
          <input
            value={settings.model}
            onChange={(event) => setSettings((current) => ({ ...current, model: event.target.value }))}
            placeholder="gpt-4o-mini"
            disabled={loading}
          />
        </label>

        {error ? <div className="error-line">{error}</div> : null}
        {message ? <div className="success-line">{message}</div> : null}

        <div className="settings-actions">
          <button className="save-button" type="submit" disabled={loading || saving}>
            <Save size={18} />
            {saving ? "保存中..." : "保存设置"}
          </button>
        </div>
      </form>
    </section>
  );
}

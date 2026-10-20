/* No microphone audio is stored by this monitor. */
(() => {
  const host = document.querySelector('#operate > div');
  const card = document.createElement('section');
  card.className = 'mission';

  card.innerHTML = `
    <h2>SEE / HEAR · Live observation</h2>

    <div class="row">
      <button id="hearToggle">Enable HEAR</button>
      <span id="hearMode">Connecting…</span>
    </div>

    <p id="missionSignal" role="status">
      Waiting for fresh data
    </p>

    <div class="kv">
      <span>Camera evidence</span>
      <b id="seeEvidence">—</b>

      <span>Speech score</span>
      <b id="hearScore">—</b>

      <span>Last speech (local time)</span>
      <b id="hearLast">—</b>
    </div>

    <p><small>
      Speech is a cue for operator inspection,
      not confirmed human presence.
      Disable HEAR before microphone tests or pilot recordings.
      Talk playback is flagged approximately;
      echo cancellation is not available.
    </small></p>

    <div id="hearError" role="alert"></div>
  `;

  host.prepend(card);

  const el = id => document.getElementById(id);
  let enabled = false;
  let changing = false;

  const labels = {
    both: 'Person visible + speech detected',
    person: 'Person visible',
    voice: 'Speech detected — inspect the scene',
    none: 'No current person / speech cue'
  };

  async function api(options) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 2500);

    try {
      const response = await fetch('/api/monitor', {
        ...options,
        signal: ctrl.signal,
        cache: 'no-store'
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Monitor request failed');
      }

      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  el('hearToggle').onclick = async () => {
    changing = true;
    el('hearToggle').disabled = true;

    try {
      await api({
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled: !enabled})
      });

      el('hearError').textContent = '';
    } catch (e) {
      el('hearError').textContent = e.message;
    } finally {
      changing = false;
    }
  };

  async function poll() {
    try {
      const s = await api();

      enabled = [
        'starting', 'listening', 'stopping'
      ].includes(s.mode);

      el('hearToggle').textContent = enabled
        ? 'Disable HEAR'
        : 'Enable HEAR';

      el('hearToggle').disabled =
        changing || s.mode === 'stopping';

      el('hearMode').textContent =
        s.mode === 'listening' && !s.fresh
          ? 'Waiting for fresh audio'
          : s.mode;

      el('hearScore').textContent =
        s.fresh && s.score !== null
          ? s.score.toFixed(3)
          : '—';

      el('hearLast').textContent = s.last_voice
        ? new Date(s.last_voice).toLocaleTimeString()
        : '—';

      el('seeEvidence').textContent = !s.see_fresh
        ? 'Off / stale'
        : s.person
          ? 'Person detected'
          : 'No person detected';

      el('missionSignal').textContent =
        labels[s.signal] +
        (s.playback && s.fresh
          ? ' · Robot playback possible'
          : '');

      card.dataset.signal = s.signal;

      if (s.error) {
        el('hearError').textContent = s.error;
      }
    } catch (e) {
      el('hearMode').textContent = 'Connection unavailable';
      el('missionSignal').textContent = 'Live status unavailable';
      el('seeEvidence').textContent = 'Unknown';
      el('hearScore').textContent = '—';
      el('hearToggle').disabled = true;
      card.dataset.signal = 'none';
    } finally {
      setTimeout(poll, 500);
    }
  }

  poll();
})();
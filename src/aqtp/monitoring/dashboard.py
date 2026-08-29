"""Local monitoring dashboard (REQ 60).

Serves a single self-contained page plus a JSON endpoint, bound to localhost by
default. It is read-only: the dashboard can display the kill-switch state but
cannot engage one — control actions belong in the CLI, where they are logged and
attributable.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Callable

from ..configuration.schema import MonitoringConfig
from ..core.logging import get_logger

logger = get_logger(__name__)

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AQTP Monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0f1115;--panel:#171a21;--line:#252a34;--fg:#e6e8eb;--muted:#8b93a1;
       --ok:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
        align-items:center;gap:16px;flex-wrap:wrap}
 .mode{font-weight:700;letter-spacing:.08em;padding:4px 12px;border-radius:4px}
 .mode.PAPER{background:#1f3a5f;color:#79c0ff}
 .mode.DEMO{background:#4a3a12;color:#e3b341}
 .mode.LIVE{background:#5f1f1f;color:#ff7b72;animation:pulse 2s infinite}
 .mode.BACKTEST{background:#2a2a2a;color:var(--muted)}
 @keyframes pulse{50%{opacity:.6}}
 main{padding:20px;display:grid;gap:16px;
      grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
 .card h2{margin:0 0 10px;font-size:12px;text-transform:uppercase;
          letter-spacing:.1em;color:var(--muted)}
 table{width:100%;border-collapse:collapse}
 td{padding:3px 0;vertical-align:top}
 td:first-child{color:var(--muted);padding-right:12px;white-space:nowrap}
 td:last-child{text-align:right}
 .wide{grid-column:1/-1;overflow-x:auto}
 .wide table{min-width:600px}
 .wide td,.wide th{text-align:left;padding:4px 10px 4px 0;border-bottom:1px solid var(--line)}
 .wide th{color:var(--muted);font-weight:400;font-size:12px}
 .ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
 .num{font-variant-numeric:tabular-nums}
 footer{padding:10px 20px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
</style></head><body>
<header>
  <span class="mode" id="mode">…</span>
  <span id="run" style="color:var(--muted)"></span>
  <span id="allowed"></span>
</header>
<main id="root"></main>
<footer>refreshes every 3s · <span id="ts"></span></footer>
<script>
const fmt=n=>n==null?'—':(typeof n==='number'?n.toLocaleString(undefined,{maximumFractionDigits:2}):n);
const cls=(v,good)=>v==null?'':(good(v)?'ok':'bad');
function kv(title,rows,wide){
  const body=rows.map(([k,v,c])=>`<tr><td>${k}</td><td class="num ${c||''}">${v}</td></tr>`).join('');
  return `<section class="card ${wide?'wide':''}"><h2>${title}</h2><table>${body}</table></section>`;
}
function table(title,cols,rows){
  if(!rows.length)return `<section class="card wide"><h2>${title}</h2>
    <div style="color:var(--muted)">none</div></section>`;
  const head=cols.map(c=>`<th>${c}</th>`).join('');
  const body=rows.map(r=>`<tr>${r.map(c=>`<td class="num">${c}</td>`).join('')}</tr>`).join('');
  return `<section class="card wide"><h2>${title}</h2>
    <table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></section>`;
}
async function tick(){
  let d; try{ d=await (await fetch('/api/status')).json(); }
  catch(e){ document.getElementById('ts').textContent='disconnected'; return; }
  const m=document.getElementById('mode');
  m.textContent='TRADING MODE: '+d.mode; m.className='mode '+d.mode;
  document.getElementById('run').textContent='run '+(d.run_id||'—');
  const a=document.getElementById('allowed');
  a.textContent=d.trading_allowed?'● trading enabled':'● entries blocked';
  a.className=d.trading_allowed?'ok':'warn';

  const r=d.risk||{}, pe=d.prediction_engine||{}, md=d.market_data||{};
  const parts=[
    kv('Account',[
      ['equity',fmt(d.equity)],
      ['unrealized',fmt(d.unrealized_pnl),cls(d.unrealized_pnl,v=>v>=0)],
      ['open positions',fmt(d.open_positions)],
      ['universe',fmt(d.universe_size)],
    ]),
    kv('Regime',[
      ['dominant',d.regime||'—'],
      ['confidence',fmt(d.regime_confidence)],
    ]),
    kv('Risk',[
      ['drawdown level',r.drawdown_level||'—',r.drawdown_level==='NORMAL'?'ok':'bad'],
      ['drawdown',(100*(r.drawdown_pct||0)).toFixed(2)+'%'],
      ['daily P&L',fmt(r.daily_pnl),cls(r.daily_pnl,v=>v>=0)],
      ['consecutive losses',fmt(r.consecutive_losses)],
      ['trades today',fmt(r.trades_today)+' / '+fmt((r.trades_today||0)+(r.trades_remaining||0))],
      ['risk multiplier',fmt(r.risk_multiplier)],
      ['kill switches',(r.kill_switches||[]).length,(r.kill_switches||[]).length?'bad':'ok'],
    ]),
    kv('Model',[
      ['ready',pe.ready?'yes':'no',pe.ready?'ok':'warn'],
      ['degraded',pe.degraded?'YES':'no',pe.degraded?'bad':'ok'],
      ['reason',pe.degraded_reason||'—'],
      ['horizon',fmt(pe.horizon_minutes)+'m'],
      ['feature version',pe.feature_version||'—'],
    ]),
    kv('Data / API',[
      ['tracked symbols',fmt(md.tracked_symbols)],
      ['circuit broken',(md.circuit_broken||[]).length,(md.circuit_broken||[]).length?'bad':'ok'],
      ['cached chains',fmt(md.cached_chains)],
      ['last cycle',d.last_cycle||'—'],
      ['reconciliation',d.last_reconciliation||'—'],
    ]),
  ];

  const lat=d.latency||{};
  const latRows=Object.entries(lat).map(([k,v])=>[k,v.count,fmt(v.median),fmt(v.p95),fmt(v.max)]);
  parts.push(table('Execution latency (ms)',['metric','n','median','p95','max'],latRows));

  const sh=(d.strategy_health||[]).map(s=>[s.strategy,s.state,s.trades,
    (100*(s.win_rate||0)).toFixed(0)+'%',fmt(s.expectancy_r),s.consecutive_losses,s.note||'']);
  parts.push(table('Strategy health',
    ['strategy','state','trades','win','expectancy R','streak','note'],sh));

  const ops=(d.opportunities||[]).map(o=>[o.underlying,o.decision,fmt(o.score),o.reason||'']);
  parts.push(table('Active opportunities',['underlying','decision','score','note'],ops));

  const pos=(d.positions||[]).map(p=>[p.instrument,p.strategy,p.direction,p.quantity,
    fmt(p.entry),fmt(p.last),fmt(p.stop),fmt(p.target),fmt(p.unrealized),fmt(p.r)]);
  parts.push(table('Open positions',
    ['instrument','strategy','dir','qty','entry','last','stop','target','P&L','R'],pos));

  const dec=(d.recent_decisions||[]).map(x=>[x.timestamp?x.timestamp.slice(11,19):'',
    x.underlying,x.decision,x.approved?'approved':'rejected',x.rejection_reason||'']);
  parts.push(table('Recent decisions',['time','underlying','decision','status','reason'],dec));

  document.getElementById('root').innerHTML=parts.join('');
  document.getElementById('ts').textContent=new Date().toLocaleTimeString();
}
tick(); setInterval(tick,3000);
</script></body></html>"""


class Dashboard:
    """Local dashboard backed by the orchestrator's status."""

    def __init__(
        self,
        config: MonitoringConfig,
        status_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self.config = config
        self.status_provider = status_provider
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def build_app(self):
        """FastAPI app, imported lazily so the dependency stays optional."""
        try:
            from fastapi import FastAPI
            from fastapi.responses import HTMLResponse, JSONResponse
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the dashboard needs fastapi and uvicorn: pip install 'aqtp[dashboard]'"
            ) from exc

        app = FastAPI(title="AQTP Monitor", docs_url=None, redoc_url=None)

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return _PAGE

        @app.get("/api/status")
        def status() -> Any:
            try:
                return JSONResponse(self.status_provider())
            except Exception as exc:
                logger.error("dashboard status failed: %s", exc)
                return JSONResponse({"error": str(exc)}, status_code=500)

        @app.get("/api/health")
        def health() -> Any:
            return {"ok": True, "timestamp": datetime.now().isoformat()}

        return app

    def start(self) -> None:
        """Run the dashboard on a daemon thread so it never blocks trading."""
        if not self.config.dashboard_enabled:
            return
        try:
            import uvicorn
        except ImportError:  # pragma: no cover
            logger.warning("uvicorn is not installed; dashboard disabled")
            return

        app = self.build_app()
        config = uvicorn.Config(
            app,
            host=self.config.dashboard_host,
            port=self.config.dashboard_port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="dashboard")
        self._thread.start()
        logger.warning(
            "dashboard: http://%s:%d", self.config.dashboard_host, self.config.dashboard_port
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

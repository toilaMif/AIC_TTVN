import { useEffect, useMemo, useRef, useState } from 'react';

const MODES = [['visual', 'Visual / Text'], ['asr', 'ASR'], ['ocr', 'OCR'], ['object', 'Object']];

function formatTimestamp(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(2).padStart(5, '0')}`;
}

function QueryInput({ item, canRemove, onChange, onRemove }) {
  return <section className="query-card">
    <div className="modality-tabs">{MODES.map(([value, label]) => {
      const disabled = value === 'ocr' || value === 'object';
      return <label className={disabled ? 'disabled' : ''} key={value}><input type="radio" disabled={disabled} checked={item.modality === value} onChange={() => onChange({ modality: value })}/><span>{label}</span></label>;
    })}</div>
    <textarea value={item.text} onChange={event => onChange({ text: event.target.value })} placeholder={item.modality === 'asr' ? 'Nhập nội dung lời nói cần tìm...' : 'Mô tả cảnh cần tìm...'}/>
    <footer><button disabled={!canRemove} onClick={onRemove}>- Remove input</button></footer>
  </section>;
}

function ResultCard({ item, index, modality, onOpen }) {
  return <article className="result-card" role="button" tabIndex="0" onClick={() => onOpen(item, modality)} onKeyDown={event => (event.key === 'Enter' || event.key === ' ') && onOpen(item, modality)}>
    <span className="rank">{index + 1}</span>
    {item.frame_url ? <img src={item.frame_url} loading="lazy"/> : <div className="image-missing">No frame preview</div>}
    <div className="result-body"><strong>{item.video_id}</strong>
      {modality === 'asr' ? <><p>{item.text}</p><small>{Number(item.start_time || 0).toFixed(2)}s - {Number(item.end_time || 0).toFixed(2)}s</small></> : <small>frame {item.frame_idx} · {Number(item.pts_time || 0).toFixed(2)}s · score {Number(item.score || 0).toFixed(4)}</small>}
    </div>
  </article>;
}

function PlayerScreen() {
  const [data, setData] = useState(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [workspaceWidth, setWorkspaceWidth] = useState(35);
  const iframeRef = useRef(null);
  const currentTimeRef = useRef(0);
  const ytPlayerRef = useRef(null);
  useEffect(() => {
    try {
      const encoded = new URLSearchParams(window.location.search).get('data');
      setData(encoded ? JSON.parse(decodeURIComponent(encoded)) : JSON.parse(sessionStorage.getItem('aic-player') || 'null'));
    } catch { setData(null); }
  }, []);
  useEffect(() => {
    const receive = event => { if (typeof event.data !== 'string') return; try { const message = JSON.parse(event.data); if (message.event === 'infoDelivery' && Number.isFinite(message.info?.currentTime)) currentTimeRef.current = message.info.currentTime; } catch {} };
    window.addEventListener('message', receive); return () => window.removeEventListener('message', receive);
  }, []);
  useEffect(() => {
    const load = () => { if (!window.YT || !iframeRef.current) return; ytPlayerRef.current = new window.YT.Player(iframeRef.current, { events: { onReady: e => e.target.playVideo(), onStateChange: e => setPlaying(e.data === window.YT.PlayerState.PLAYING) } }); };
    if (window.YT?.Player) load(); else { const script = document.createElement('script'); script.src = 'https://www.youtube.com/iframe_api'; document.body.appendChild(script); window.onYouTubeIframeAPIReady = load; }
    return () => { window.onYouTubeIframeAPIReady = null; };
  }, [data]);
  const seek = seconds => { const player = ytPlayerRef.current; if (player?.getCurrentTime) player.seekTo(Math.max(0, player.getCurrentTime() + seconds), true); };
  const togglePlayback = () => { const player = ytPlayerRef.current; if (!player) return; playing ? player.pauseVideo() : player.playVideo(); };
  const changeSpeed = value => { ytPlayerRef.current?.setPlaybackRate?.(value); setSpeed(value); };
  const closePlayer = () => {
    window.close();
    // Browsers refuse window.close() for tabs they did not open themselves.
    window.setTimeout(() => window.history.back(), 50);
  };
  const resizeWorkspace = event => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = workspaceWidth;
    const layoutWidth = event.currentTarget.parentElement?.getBoundingClientRect().width || window.innerWidth;
    const move = pointer => setWorkspaceWidth(Math.max(20, Math.min(55, startWidth - ((pointer.clientX - startX) / layoutWidth) * 100)));
    const stop = () => {
      document.body.classList.remove('resizing');
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
    };
    document.body.classList.add('resizing');
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
  };
  if (!data) return <main className="player-screen"><h1>Khong co du lieu keyframe</h1><button className="close-player" onClick={closePlayer} aria-label="Dong cua so">&times;</button></main>;
  const { item, modality, watchUrl } = data;
  const timestamp = Math.max(0, Number(modality === 'asr' ? item.start_time : item.pts_time) || 0);
  const start = Math.floor(timestamp);
  const match = watchUrl.match(/[?&]v=([^&]+)/); const embed = match ? `https://www.youtube.com/embed/${match[1]}?start=${start}&autoplay=1&enablejsapi=1` : watchUrl;
  return <main className="player-screen"><header className="player-header"><div className="player-title"><h1>Frame Player</h1><p>{item.video_id} - {modality.toUpperCase()}</p></div><div className="keyframe-summary">{item.frame_url && <img src={item.frame_url} alt="Shot preview" />}<div><span>Video</span><b>{item.video_id}</b></div><div><span>Keyframe ID</span><b>{item.keyframe_id || '-'}</b></div><div><span>Frame</span><b>{item.frame_idx ?? '-'}</b></div><div><span>Timestamp</span><b>{timestamp.toFixed(2)}s ({formatTimestamp(timestamp)})</b></div><div><span>Shot</span><b>{item.shot_id || '-'}</b></div><div><span>Score</span><b>{item.score != null ? Number(item.score).toFixed(4) : '-'}</b></div>{modality === 'asr' && <div className="summary-text"><span>ASR</span><b>{item.text}</b></div>}</div><button className="close-player" onClick={closePlayer} aria-label="Dong cua so" title="Dong cua so">&times;</button></header><div className="player-layout" style={{ '--workspace-width': `${workspaceWidth}%` }}><div className="player-main"><div className="player-video"><iframe ref={iframeRef} title="Video nguon" src={embed} allow="autoplay; encrypted-media" allowFullScreen /></div><div className="player-controls"><button onClick={() => seek(-10)}>↶<small>10</small></button><button title="Lui 5 giay" onClick={() => seek(-5)}>↶<small>5</small></button><button className="play-control" onClick={togglePlayback}>{playing ? "⏸" : "▶"}</button><button title="Tien 5 giay" onClick={() => seek(5)}>↷<small>5</small></button><button title="Tien 10 giay" onClick={() => seek(10)}>↷<small>10</small></button><span className="control-divider" />{[0.5,1,1.5,2].map(value => <button className={speed === value ? "active" : ""} onClick={() => changeSpeed(value)} key={value}>x{value}</button>)}</div></div><div className="workspace-splitter" onPointerDown={resizeWorkspace} onDoubleClick={() => setWorkspaceWidth(35)} role="separator" aria-label="Resize workspace" aria-orientation="vertical" aria-valuemin="20" aria-valuemax="55" aria-valuenow={Math.round(workspaceWidth)} /><aside className="player-workspace" aria-label="Workspace" /></div></main>;
}

export default function App() {
  if (new URLSearchParams(window.location.search).get('player') === '1') return <PlayerScreen />;
  const [inputs, setInputs] = useState([{ id: 1, modality: 'visual', text: '' }, { id: 2, modality: 'asr', text: '' }]);
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const [searchMode, setSearchMode] = useState('whole');
  const [boxText, setBoxText] = useState('');
  const [boxes, setBoxes] = useState([{ id: 1, text: 'diver', x: 12, y: 16, w: 62, h: 34 }]);
  const [selectedBox, setSelectedBox] = useState(1);
  const [sidebarWidth, setSidebarWidth] = useState(420);
  const [visualWidth, setVisualWidth] = useState(50);
  const [config, setConfig] = useState({ topK: 20, candidateK: 200, temporalGap: 3 });
  const [sceneFilter, setSceneFilter] = useState('all');
  const [displayView, setDisplayView] = useState('grid');
  const [resultModality, setResultModality] = useState('visual');
  const [player, setPlayer] = useState(null);

  const total = useMemo(() => groups.reduce((sum, group) => sum + group.items.length, 0), [groups]);
  const visibleGroups = useMemo(() => groups.filter(group => group.modality === resultModality), [groups, resultModality]);
  const visibleTotal = useMemo(() => visibleGroups.reduce((sum, group) => sum + group.items.length, 0), [visibleGroups]);
  const activeTitle = inputs.filter(item => item.text.trim()).map(item => item.text.trim()).join(' + ') || 'Multimodal video retrieval';

  useEffect(() => {
    const handler = event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') search(); };
    window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler);
  });

  const patchInput = (id, patch) => setInputs(items => items.map(item => item.id === id ? { ...item, ...patch } : item));
  const addInput = () => setInputs(items => [...items, { id: Date.now(), modality: 'visual', text: '' }]);
  const addBox = () => { if (!boxText.trim()) return; const id = Date.now(); setBoxes(items => [...items, { id, text: boxText.trim(), x: 12 + items.length * 6, y: 15 + items.length * 5, w: 42, h: 28 }]); setSelectedBox(id); setBoxText(''); };

  async function openPlayer(item, modality) {
    const response = await fetch(`/videos/${encodeURIComponent(item.video_id)}`);
    if (!response.ok) return setError('Không tìm thấy URL video nguồn.');
    const video = await response.json();
    const start = Math.max(0, Math.floor(modality === 'asr' ? item.start_time : item.pts_time || 0));
    sessionStorage.setItem('aic-player', JSON.stringify({ item, modality, watchUrl: video.watch_url }));
    const payload = encodeURIComponent(JSON.stringify({ item, modality, watchUrl: video.watch_url }));
    window.open(`${window.location.origin}/ui/?player=1&data=${payload}`, '_blank', 'noopener,noreferrer');
  }

  function resizePanel(event, type) {
    event.preventDefault();
    const startX = event.clientX;
    const startSidebar = sidebarWidth;
    const startVisual = visualWidth;
    const move = pointer => {
      const dx = pointer.clientX - startX;
      if (type === 'sidebar') setSidebarWidth(Math.max(300, Math.min(620, startSidebar + dx)));
      else setVisualWidth(Math.max(25, Math.min(75, startVisual + (dx / Math.max(window.innerWidth - sidebarWidth, 1)) * 100)));
    };
    const stop = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', stop); document.body.classList.remove('resizing'); };
    document.body.classList.add('resizing'); window.addEventListener('pointermove', move); window.addEventListener('pointerup', stop);
  }

  function beginBoxInteraction(event, id, action) {
    event.preventDefault(); event.stopPropagation(); setSelectedBox(id);
    const canvas = event.currentTarget.closest('.spatial-canvas').getBoundingClientRect();
    const box = boxes.find(item => item.id === id); if (!box) return;
    const start = { px: event.clientX, py: event.clientY, ...box };
    const move = pointer => {
      const dx = ((pointer.clientX - start.px) / canvas.width) * 100;
      const dy = ((pointer.clientY - start.py) / canvas.height) * 100;
      setBoxes(items => items.map(item => {
        if (item.id !== id) return item;
        if (action === 'move') return { ...item, x: Math.max(0, Math.min(100 - item.w, start.x + dx)), y: Math.max(0, Math.min(100 - item.h, start.y + dy)) };
        return { ...item, w: Math.max(12, Math.min(100 - item.x, start.w + dx)), h: Math.max(12, Math.min(100 - item.y, start.h + dy)) };
      }));
    };
    const stop = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', stop); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', stop);
  }

  async function search() {
    const active = inputs.filter(item => item.text.trim() && ['visual', 'asr'].includes(item.modality));
    if (!active.length || loading) return setError('Hãy nhập ít nhất một truy vấn Visual hoặc ASR.');
    setLoading(true); setGroups([]); setError(''); setElapsed(0);
    const started = performance.now(); const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 100);
    try {
      const results = await Promise.all(active.map(async item => {
        const candidateParam = item.modality === 'visual' ? `&candidate_k=${config.candidateK}` : '';
        const response = await fetch(`/search/${item.modality}?query=${encodeURIComponent(item.text.trim())}&top_k=${config.topK}${candidateParam}`);
        if (!response.ok) throw new Error(await response.text());
        return { modality: item.modality, query: item.text.trim(), items: await response.json() };
      }));
      setGroups(results);
      if (results.length) setResultModality(results[0].modality);
    } catch (exception) { setError(exception.message); }
    finally { clearInterval(timer); setElapsed((performance.now() - started) / 1000); setLoading(false); }
  }

  return <div className="app-shell" style={{ '--sidebar-width': `${sidebarWidth}px` }}>
    <aside className="sidebar">
      <header className="brand"><img src="/assets/LogoAIC_TTVN_1.png"/><div><strong>Video Search AIC 2026</strong><span className="brand-tagline"><i className="tt">TT</i><i className="v">V</i><i className="n">N</i><em> - IUH - one day for all</em></span></div></header>
      <div className="section-heading"><span>QUERY INPUTS</span><button className="add-input-button" onClick={addInput}>+ ADD INPUT</button></div>
      <div className="query-list">{inputs.map(item => <QueryInput key={item.id} item={item} canRemove={inputs.length > 1} onChange={patch => patchInput(item.id, patch)} onRemove={() => setInputs(items => items.filter(value => value.id !== item.id))}/>)}</div>
      <button className="search-button" disabled={loading} onClick={search}>{loading ? <><i className="mini-spinner"/> Searching {elapsed.toFixed(1)}s</> : <>Search <kbd>Ctrl</kbd> <kbd>Enter</kbd></>}</button>
      {error && <p className="error-banner">{error}</p>}
      <section className="config-section"><div className="section-heading"><span>CONFIGURATIONS</span></div><div className="config-grid"><label>Results<input type="number" min="1" max="100" value={config.topK} onChange={event => setConfig({...config, topK: Math.max(1, Math.min(100, Number(event.target.value) || 1))})}/></label><label>Candidate K<input type="number" min="20" max="500" value={config.candidateK} onChange={event => setConfig({...config, candidateK: Math.max(20, Math.min(500, Number(event.target.value) || 20))})}/></label><label>Time gap (s)<input type="number" min="0" max="60" step="0.5" value={config.temporalGap} onChange={event => setConfig({...config, temporalGap: Math.max(0, Math.min(60, Number(event.target.value) || 0))})}/></label></div></section>
      <section className="mode-section"><div className="section-heading"><span>SEARCH MODE</span></div>{[['whole','Whole image'],['localized','Localized'],['conjunctive','Conjunctive']].map(([key,label]) => <button key={key} onClick={() => setSearchMode(key)} className={searchMode === key ? 'mode-button active' : 'mode-button'}>{label}</button>)}</section>
      {searchMode !== 'whole' && <section className="local-section"><div className="section-heading"><span>LOCAL QUERY BOXES</span><button onClick={() => { setBoxes([]); setSelectedBox(null); }}>CLEAR</button></div><div className="add-box"><input value={boxText} onChange={event => setBoxText(event.target.value)} onKeyDown={event => event.key === 'Enter' && addBox()} placeholder="e.g. red car"/><button onClick={addBox}>Add box</button></div><div className="box-list">{boxes.map((box,index) => <div className={selectedBox === box.id ? 'box-item selected' : 'box-item'} key={box.id}><input value={box.text} onFocus={() => setSelectedBox(box.id)} onChange={event => setBoxes(items => items.map(item => item.id === box.id ? {...item,text:event.target.value} : item))}/><button onClick={() => { setBoxes(items => items.filter(item => item.id !== box.id)); if(selectedBox === box.id) setSelectedBox(boxes.find(item => item.id !== box.id)?.id || null); }}>×</button></div>)}</div><div className="spatial-canvas">{boxes.map((box,index) => <div onClick={() => setSelectedBox(box.id)} onPointerDown={event => beginBoxInteraction(event, box.id, 'move')} className={`${selectedBox === box.id ? 'spatial-box selected-box' : 'spatial-box'} color-${index % 3}`} key={box.id} style={{left:`${box.x}%`,top:`${box.y}%`,width:`${box.w}%`,height:`${box.h}%`}}><b>{box.text}</b><span>{(box.x/100).toFixed(2)}, {(box.y/100).toFixed(2)}, {((box.x+box.w)/100).toFixed(2)}, {((box.y+box.h)/100).toFixed(2)}</span><i className="resize-handle" onPointerDown={event => beginBoxInteraction(event, box.id, 'resize')}/></div>)}</div></section>}
    </aside><div className="panel-splitter sidebar-splitter" onPointerDown={event => resizePanel(event, 'sidebar')} />
    <main className="results-pane">
      <header className="results-header"><div><span className="results-label">RESULTS <b>{loading ? `${elapsed.toFixed(1)}s` : visibleTotal}</b></span><h1>{activeTitle}</h1><div className="result-summary"><small>{searchMode.toUpperCase()}</small></div></div><div className="result-meta"><div className="result-type-switcher">{[['visual','Visual'],['ocr','OCR'],['asr','ASR'],['object','Object']].map(([value,label]) => <button type="button" key={value} className={resultModality === value ? 'active' : ''} onClick={() => setResultModality(value)}>{label}</button>)}</div><select className="view-select" aria-label="Result view" value={displayView} onChange={event => setDisplayView(event.target.value)}><option value="grid">Grid View</option><option value="cluster">Cluster View</option><option value="similarity">Similarity View</option></select><select aria-label="Filter scene type" value={sceneFilter} onChange={event => setSceneFilter(event.target.value)}><option value="all">All scenes</option><option value="day">Daytime</option><option value="night">Night</option><option value="rain">Rain</option><option value="sunny">Sunny</option><option value="indoor">Indoor</option><option value="outdoor">Outdoor</option></select></div></header>
      <div className="notice">Visual và ASR đang hoạt động. OCR và Object sẽ được mở khi index tương ứng sẵn sàng.</div>
      {loading && <div className="loading-state"><i className="spinner"/><strong>Đang tìm kiếm video</strong><span>Querying indexes and ranking candidates...</span><time>{elapsed.toFixed(1)}s</time></div>}
      {!loading && !groups.length && <div className="welcome"><strong>Bắt đầu bằng một truy vấn</strong><p>Nhập mô tả hình ảnh hoặc lời thoại tiếng Việt ở cột bên trái.</p></div>}
      {!loading && groups.length > 0 && visibleGroups.length === 0 && <div className="empty-state">No {resultModality.toUpperCase()} results available.</div>}
      {!loading && visibleGroups.length > 0 && <div className={`result-columns view-${displayView}`}>{visibleGroups.map((group,index) => <><section className="result-group" style={{width: '100%'}} key={`${group.modality}-${group.query}`}><header><h2>{group.modality === 'asr' ? 'ASR transcript' : 'Visual / Text'}</h2><span>{group.items.length}</span></header><p className="group-query">“{group.query}”</p><div className="result-grid">{group.items.length ? group.items.map((item,index) => <ResultCard key={item.keyframe_id || item.segment_id} item={item} index={index} modality={group.modality} onOpen={openPlayer}/>) : <div className="empty-state">Không tìm thấy kết quả phù hợp.</div>}</div></section></>)}</div>}
      {player && <div className="player-backdrop" onMouseDown={event => event.target === event.currentTarget && setPlayer(null)}><section className="player-modal"><header><div><strong>{player.video_id}</strong><span>{player.start}s</span></div><button aria-label="Close video" onClick={() => setPlayer(null)}>×</button></header>{player.embedUrl ? <iframe src={player.embedUrl} title={player.video_id} allow="autoplay; encrypted-media; picture-in-picture" allowFullScreen/> : <img src={player.frame_url}/>}<footer><a href={player.watchUrl} target="_blank" rel="noreferrer">Open video at {player.start}s</a></footer></section></div>}
    </main>
  </div>;
}

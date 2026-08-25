import { useEffect, useMemo, useRef, useState } from 'react';

const AUTH_STORAGE_KEY = 'aic-auth-token';

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = localStorage.getItem(AUTH_STORAGE_KEY);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    window.dispatchEvent(new Event('aic-auth-expired'));
  }
  return response;
}

function logout() {
  localStorage.removeItem(AUTH_STORAGE_KEY);
  window.dispatchEvent(new Event('aic-auth-expired'));
}

const ACTIVE_QUESTION_KEY = 'aic-active-question';

function setActiveQuestion(item) {
  localStorage.setItem(ACTIVE_QUESTION_KEY, JSON.stringify({
    question_id: item.question_id,
    part: item.part,
    number: item.number,
    qtype: item.qtype,
    eventsCount: item.events?.length || 0,
  }));
}

function getActiveQuestion() {
  try {
    return JSON.parse(localStorage.getItem(ACTIVE_QUESTION_KEY) || 'null');
  } catch {
    return null;
  }
}

const AVATAR_COLORS = ['#ef3d00', '#2657c9', '#1f8a3d', '#6a2fc9', '#b5690a', '#0d8f9e', '#a3226f'];

function avatarColor(username) {
  if (!username) return AVATAR_COLORS[0];
  let hash = 0;
  for (let index = 0; index < username.length; index += 1) hash = (hash * 31 + username.charCodeAt(index)) >>> 0;
  return AVATAR_COLORS[hash % AVATAR_COLORS.length];
}

function avatarInitials(username) {
  if (!username) return '?';
  const name = username.includes('_') ? username.split('_').slice(1).join('_') : username;
  return (name || username).slice(0, 2).toUpperCase();
}

function Avatar({ username, title, size }) {
  const [imageFailed, setImageFailed] = useState(false);
  const className = size === 'lg' ? 'avatar avatar-lg' : 'avatar';
  if (username && !imageFailed) {
    return <img
      className={`${className} avatar-image`}
      src={`${import.meta.env.BASE_URL}avatar/${encodeURIComponent(username)}.png`}
      alt={username}
      title={title || username}
      onError={() => setImageFailed(true)}
    />;
  }
  return <span className={className} style={{ background: avatarColor(username) }} title={title || username || ''}>{avatarInitials(username)}</span>;
}

function useCurrentAccount() {
  const [account, setAccount] = useState({ username: null, isAdmin: false });
  useEffect(() => {
    let cancelled = false;
    apiFetch('/auth/me')
      .then(response => response.ok ? response.json() : null)
      .then(data => { if (!cancelled && data) setAccount({ username: data.username, isAdmin: !!data.is_admin }); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);
  return account;
}

const MODES = [['visual', 'Visual / Text'], ['asr', 'ASR'], ['ocr', 'OCR'], ['object', 'Object'], ['topic', 'Topic']];

function LockIcon({ locked }) {
  return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="4" y="11" width="16" height="10" rx="2"/>
    {locked ? <path d="M8 11V7a4 4 0 0 1 8 0v4"/> : <path d="M8 11V7a4 4 0 0 1 7.75-1.5"/>}
  </svg>;
}

function formatTimestamp(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(2).padStart(5, '0')}`;
}

const FUSION_RRF_K = 60;

function clusterKey(item) {
  return item.shot_id ? `shot:${item.shot_id}` : `video:${item.video_id}`;
}

function groupByVideo(items) {
  // One row per source video, rows kept in order of first appearance — since
  // items already arrive ranked by relevance, that is the same as ordering
  // rows by each video's best-ranked frame (PraK V4's grouped keyframe layout).
  const order = [];
  const groups = new Map();
  items.forEach(item => {
    const key = item.video_id;
    if (!groups.has(key)) { groups.set(key, []); order.push(key); }
    groups.get(key).push(item);
  });
  return order.map(key => ({ key, items: groups.get(key) }));
}

function fuseResults(groups) {
  if (groups.length < 2) return [];
  const merged = new Map();
  groups.forEach(group => {
    group.items.forEach((item, index) => {
      const key = clusterKey(item);
      const contribution = 1 / (FUSION_RRF_K + index + 1);
      const entry = merged.get(key) || { item, itemModality: group.modality, fusionScore: 0, contributions: [] };
      entry.fusionScore += contribution;
      entry.contributions.push({ modality: group.modality, query: group.query, rank: index + 1, score: item.score });
      if (!entry.item.frame_url && item.frame_url) { entry.item = item; entry.itemModality = group.modality; }
      merged.set(key, entry);
    });
  });
  return Array.from(merged.entries())
    .map(([key, value]) => ({ key, ...value }))
    .sort((a, b) => b.fusionScore - a.fusionScore);
}

function QueryInput({ item, canRemove, onChange, onRemove }) {
  return <section className="query-card">
    <div className="modality-tabs">{MODES.map(([value, label]) => {
      return <label key={value}><input type="radio" checked={item.modality === value} onChange={() => onChange({ modality: value })}/><span>{label}</span></label>;
    })}</div>
    <textarea value={item.text} onChange={event => onChange({ text: event.target.value })} placeholder={item.modality === 'asr' ? 'Nhập nội dung lời nói cần tìm...' : item.modality === 'ocr' ? 'Nhập chữ xuất hiện trong video...' : item.modality === 'object' ? 'Nhập tên vật thể cần tìm...' : item.modality === 'topic' ? 'Nhập chủ đề/nội dung tổng quát của video...' : 'Mô tả cảnh cần tìm...'}/>
    <footer><button disabled={!canRemove} onClick={onRemove}>- Remove input</button></footer>
  </section>;
}

function FusionCard({ entry, index, onOpen }) {
  const { item, itemModality, fusionScore, contributions } = entry;
  return <article className="result-card fusion-card" role="button" tabIndex="0" onClick={() => onOpen(item, itemModality)} onKeyDown={event => (event.key === 'Enter' || event.key === ' ') && onOpen(item, itemModality)}>
    <span className="rank">{index + 1}</span>
    {item.frame_url ? <img src={item.frame_url} loading="lazy"/> : <div className="image-missing">No frame preview</div>}
    <div className="result-body"><strong>{item.video_id}</strong>
      <small>frame {item.frame_idx ?? '-'} · {Number(item.pts_time ?? item.start_time ?? 0).toFixed(2)}s · fusion {fusionScore.toFixed(4)}</small>
      <div className="fusion-badges">{contributions.map((contribution, badgeIndex) => <span className={`fusion-badge modality-${contribution.modality}`} key={badgeIndex} title={contribution.query}>{contribution.modality.toUpperCase()} #{contribution.rank}</span>)}</div>
    </div>
  </article>;
}

function ResultCard({ item, index, modality, onOpen }) {
  return <article className="result-card" role="button" tabIndex="0" onClick={() => onOpen(item, modality)} onKeyDown={event => (event.key === 'Enter' || event.key === ' ') && onOpen(item, modality)}>
    <span className="rank">{index + 1}</span>
    {item.frame_url ? <img src={item.frame_url} loading="lazy"/> : <div className="image-missing">No frame preview</div>}
    <div className="result-body"><strong>{item.video_id}</strong>
      {modality === 'asr' ? <><p>{item.text}</p><small>{Number(item.start_time || 0).toFixed(2)}s - {Number(item.end_time || 0).toFixed(2)}s</small></> : <>{item.text && <p>{item.text}</p>}<small>frame {item.frame_idx} · {Number(item.pts_time || 0).toFixed(2)}s · score {Number(item.score || 0).toFixed(4)}</small></>}
    </div>
  </article>;
}

function VideoSearch({ currentVideoId, onSelect }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const rootRef = useRef(null);

  useEffect(() => {
    const close = event => { if (!rootRef.current?.contains(event.target)) setOpen(false); };
    document.addEventListener('pointerdown', close);
    return () => document.removeEventListener('pointerdown', close);
  }, []);

  useEffect(() => {
    const term = query.trim();
    if (!term) {
      setResults([]);
      setLoading(false);
      setError('');
      setOpen(false);
      return undefined;
    }
    setResults([]);
    setActiveIndex(0);
    setError('');
    setLoading(true);
    setOpen(true);
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const response = await apiFetch(`/videos?query=${encodeURIComponent(term)}&limit=8`, { signal: controller.signal });
        if (!response.ok) throw new Error('Không thể tìm kiếm video.');
        const items = await response.json();
        setResults(items);
        setActiveIndex(0);
      } catch (exception) {
        if (exception.name !== 'AbortError') {
          setResults([]);
          setError(exception.message || 'Không thể tìm kiếm video.');
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 250);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [query]);

  const choose = video => {
    onSelect(video);
    setQuery('');
    setResults([]);
    setOpen(false);
  };
  const submit = event => {
    event.preventDefault();
    if (results[activeIndex]) choose(results[activeIndex]);
  };
  const handleKeyDown = event => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setOpen(true);
      setActiveIndex(index => Math.min(index + 1, Math.max(results.length - 1, 0)));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActiveIndex(index => Math.max(index - 1, 0));
    } else if (event.key === 'Escape') {
      setOpen(false);
    }
  };

  return <div className="video-search" ref={rootRef}>
    <form className="video-search-form" role="search" onSubmit={submit}>
      <span className="video-search-icon" aria-hidden="true">⌕</span>
      <input
        type="search"
        value={query}
        onChange={event => setQuery(event.target.value)}
        onFocus={() => query.trim() && setOpen(true)}
        onKeyDown={handleKeyDown}
        placeholder="Tìm tiêu đề hoặc ID (vd: L30_V019)"
        aria-label="Tìm video theo tiêu đề hoặc ID"
        aria-autocomplete="list"
        aria-controls="video-search-results"
        aria-expanded={open}
        autoComplete="off"
        spellCheck="false"
      />
      {loading && <i className="video-search-spinner" aria-label="Đang tìm" />}
      <button type="submit" className="video-search-submit" disabled={!results.length} aria-label="Mở video tìm được">Tìm</button>
    </form>
    {open && <div className="video-search-results" id="video-search-results" role="listbox">
      {!loading && error && <p className="video-search-message error">{error}</p>}
      {!loading && !error && !results.length && <p className="video-search-message">Không tìm thấy video phù hợp.</p>}
      {results.map((video, index) => <button
        type="button"
        role="option"
        aria-selected={index === activeIndex}
        className={`${index === activeIndex ? 'active' : ''}${video.video_id === currentVideoId ? ' current' : ''}`}
        key={video.video_id}
        onMouseEnter={() => setActiveIndex(index)}
        onClick={() => choose(video)}
      >
        {video.thumbnail_url ? <img src={video.thumbnail_url} alt="" /> : <span className="video-search-thumbnail" aria-hidden="true">▶</span>}
        <span className="video-search-result-text"><b>{video.video_id}</b><small>{video.title || 'Không có tiêu đề'}</small></span>
        {video.video_id === currentVideoId && <em>Đang phát</em>}
      </button>)}
    </div>}
  </div>;
}

const PRESENCE_STALE_MS = 5 * 60 * 1000;
const QUESTIONS_POLL_MS = 8000;

function QuestionBank({ onSelectQuestion, currentUsername, isAdmin }) {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [importing, setImporting] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [selectedId, setSelectedId] = useState(null);

  const loadQuestions = async () => {
    setError('');
    try {
      const response = await apiFetch('/questions');
      if (!response.ok) throw new Error('Không tải được danh sách câu hỏi.');
      setQuestions(await response.json());
    } catch (exception) {
      setError(exception.message || 'Không tải được danh sách câu hỏi.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadQuestions(); }, []);

  useEffect(() => {
    if (!questions.length) return undefined;
    const id = window.setInterval(loadQuestions, QUESTIONS_POLL_MS);
    return () => window.clearInterval(id);
  }, [questions.length]);

  const handleImportChange = async event => {
    const file = event.target.files[0];
    if (!file) return;
    setImporting(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await apiFetch('/questions/import', { method: 'POST', body: formData });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || 'Import bộ đề thất bại.');
      }
      setExpanded(true);
      await loadQuestions();
    } catch (exception) {
      setError(exception.message || 'Import bộ đề thất bại.');
    } finally {
      setImporting(false);
      event.target.value = '';
    }
  };

  const handleRemove = async () => {
    const setId = questions[0]?.set_id;
    if (!setId) return;
    if (!window.confirm(`Xóa bộ đề "${setId}"? Toàn bộ tiến độ đã đánh dấu sẽ mất.`)) return;
    setRemoving(true);
    setError('');
    try {
      const response = await apiFetch(`/questions/${encodeURIComponent(setId)}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Xóa bộ đề thất bại.');
      setQuestions([]);
      setSelectedId(null);
      setExpanded(false);
    } catch (exception) {
      setError(exception.message || 'Xóa bộ đề thất bại.');
    } finally {
      setRemoving(false);
    }
  };

  const toggleDone = async (questionId, done) => {
    setQuestions(items => items.map(item => item.question_id === questionId
      ? { ...item, done, done_by: done ? currentUsername : null, ...(done ? { in_progress_by: null, in_progress_at: null } : {}) }
      : item));
    try {
      await apiFetch(`/questions/${encodeURIComponent(questionId)}/done`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ done }),
      });
    } catch {
      // Optimistic update stands even if the request fails silently; a manual reload re-syncs.
    }
  };

  const saveAnswerText = async (questionId, text) => {
    setQuestions(items => items.map(item => item.question_id === questionId ? { ...item, answer_text: text } : item));
    try {
      await apiFetch(`/questions/${encodeURIComponent(questionId)}/answer/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
    } catch {
      // Best-effort; re-typing or reloading re-syncs.
    }
  };

  const selectQuestion = async item => {
    setSelectedId(item.question_id);
    setActiveQuestion(item);
    try {
      const response = await apiFetch(`/questions/${encodeURIComponent(item.question_id)}/select`, { method: 'POST' });
      if (response.ok) {
        const updated = await response.json();
        setQuestions(items => items.map(other => {
          if (other.question_id === updated.question_id) return updated;
          if (other.in_progress_by === currentUsername) return { ...other, in_progress_by: null, in_progress_at: null };
          return other;
        }));
      }
    } catch {
      // Presence marking is best-effort; the periodic poll will re-sync either way.
    }
  };

  const isPresenceActive = item => {
    if (!item.in_progress_by || !item.in_progress_at) return false;
    return Date.now() - new Date(item.in_progress_at).getTime() < PRESENCE_STALE_MS;
  };

  const doneCount = questions.filter(item => item.done).length;
  const selectedQuestion = questions.find(item => item.question_id === selectedId) || null;

  return <section className="config-section question-bank">
    {!questions.length
      ? (isAdmin && <label className={importing ? 'import-questions-button-full disabled' : 'import-questions-button-full'}>
        {importing ? 'Đang import...' : 'IMPORT BỘ ĐỀ'}
        <input type="file" accept=".zip" onChange={handleImportChange} disabled={importing} />
      </label>)
      : <button type="button" className="question-bank-toggle" onClick={() => setExpanded(value => !value)} aria-expanded={expanded}>
        <span className={expanded ? 'question-bank-chevron open' : 'question-bank-chevron'}>▾</span>
        <span>BỘ ĐỀ THI</span>
        <span className="question-bank-toggle-progress">{doneCount}/{questions.length} đã xong</span>
      </button>}
    {error && <p className="error-banner">{error}</p>}
    {loading && <p className="question-bank-message">Đang tải câu hỏi...</p>}
    {!loading && !questions.length && <p className="question-bank-message">{isAdmin ? 'Chưa có bộ đề nào được import.' : 'Chưa có bộ đề nào — chỉ admin mới import được.'}</p>}
    {!loading && questions.length > 0 && expanded && <>
      {isAdmin && <button type="button" className="remove-questions-button" onClick={handleRemove} disabled={removing}>{removing ? 'Đang xóa...' : '× Xóa bộ đề này'}</button>}
      <div className="question-list">
        {questions.map(item => {
          const present = isPresenceActive(item);
          const rowClass = item.done ? 'question-row done' : present ? 'question-row in-progress' : 'question-row';
          return <div key={item.question_id} className={rowClass} role="button" tabIndex={0} onClick={() => selectQuestion(item)} onKeyDown={event => (event.key === 'Enter' || event.key === ' ') && selectQuestion(item)}>
            <div className="question-row-main">
              <span className={`question-type-badge type-${item.qtype}`}>{item.qtype.toUpperCase()}</span>
              <span className="question-row-number">{item.part}-{item.number}</span>
              <span className="question-row-text">{item.text}</span>
              {present && <Avatar username={item.in_progress_by} title={`${item.in_progress_by} đang làm câu này`} />}
            </div>
            <div className="question-done-toggle" onClick={event => event.stopPropagation()}>
              <input type="checkbox" checked={item.done} onChange={event => toggleDone(item.question_id, event.target.checked)} />
              <span>Đã xong</span>
              {item.done && item.done_by && <Avatar username={item.done_by} title={`${item.done_by} đã đánh dấu xong`} />}
            </div>
          </div>;
        })}
      </div>
    </>}
    {selectedQuestion && <div className="question-detail">
      <div className="question-detail-head">
        <span className={`question-type-badge type-${selectedQuestion.qtype}`}>{selectedQuestion.qtype.toUpperCase()}</span>
        <span className="question-row-number">{selectedQuestion.part}-{selectedQuestion.number}</span>
      </div>
      <p className="question-detail-text">{selectedQuestion.text}</p>
      {selectedQuestion.qtype === 'qa' && <div className="question-answer-text">
        <label>Đáp án (Q&amp;A)
          <input
            type="text"
            maxLength={100}
            defaultValue={selectedQuestion.answer_text || ''}
            key={selectedQuestion.question_id}
            placeholder="Nhập câu trả lời..."
            onBlur={event => saveAnswerText(selectedQuestion.question_id, event.target.value)}
            onKeyDown={event => event.key === 'Enter' && event.target.blur()}
          />
        </label>
      </div>}
      {selectedQuestion.qtype === 'trake' && selectedQuestion.events?.length > 0 && <div className="question-events">
        {selectedQuestion.events.map(eventItem => <button type="button" key={eventItem.label} className="question-event-use" onClick={() => onSelectQuestion(eventItem.text)}>
          <b>{eventItem.label}</b><span>{eventItem.text}</span>
        </button>)}
      </div>}
    </div>}
  </section>;
}

function PlayerScreen() {
  const { username: currentUsername } = useCurrentAccount();
  const [data, setData] = useState(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [workspaceWidth, setWorkspaceWidth] = useState(35);
  const [keyframeMap, setKeyframeMap] = useState([]);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [allQuestions, setAllQuestions] = useState([]);
  const [activeQuestionId, setActiveQuestionId] = useState(() => getActiveQuestion()?.question_id || '');
  const [answerError, setAnswerError] = useState('');
  const iframeRef = useRef(null);
  const currentTimeRef = useRef(0);
  const ytPlayerRef = useRef(null);
  const activeRowRef = useRef(null);
  useEffect(() => {
    try {
      const encoded = new URLSearchParams(window.location.search).get('data');
      if (!encoded) setData(JSON.parse(sessionStorage.getItem('aic-player') || 'null'));
      else {
        try { setData(JSON.parse(encoded)); }
        catch { setData(JSON.parse(decodeURIComponent(encoded))); }
      }
    } catch { setData(null); }
  }, []);
  useEffect(() => {
    const receive = event => { if (typeof event.data !== 'string') return; try { const message = JSON.parse(event.data); if (message.event === 'infoDelivery' && Number.isFinite(message.info?.currentTime)) currentTimeRef.current = message.info.currentTime; } catch {} };
    window.addEventListener('message', receive); return () => window.removeEventListener('message', receive);
  }, []);
  useEffect(() => {
    const id = window.setInterval(() => setPlaybackTime(currentTimeRef.current), 250);
    return () => window.clearInterval(id);
  }, []);
  useEffect(() => {
    const videoId = data?.item?.video_id;
    if (!videoId) return setKeyframeMap([]);
    let cancelled = false;
    apiFetch(`/keyframes/${encodeURIComponent(videoId)}`)
      .then(response => response.ok ? response.json() : [])
      .then(rows => { if (!cancelled) setKeyframeMap(rows); })
      .catch(() => { if (!cancelled) setKeyframeMap([]); });
    return () => { cancelled = true; };
  }, [data?.item?.video_id]);
  const loadQuestions = () => {
    apiFetch('/questions')
      .then(response => response.ok ? response.json() : [])
      .then(rows => setAllQuestions(rows))
      .catch(() => {});
  };
  useEffect(() => { loadQuestions(); }, []);
  useEffect(() => {
    // Pick up "chọn câu để làm" from the BỘ ĐỀ THI tab even if this player tab was
    // already open — localStorage writes there don't touch this tab's own state.
    const onStorage = event => {
      if (event.key && event.key !== ACTIVE_QUESTION_KEY) return;
      setActiveQuestionId(getActiveQuestion()?.question_id || '');
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);
  const activeQuestion = allQuestions.find(question => question.question_id === activeQuestionId) || null;
  const changeActiveQuestion = questionId => {
    setActiveQuestionId(questionId);
    setAnswerError('');
    const question = allQuestions.find(item => item.question_id === questionId);
    if (question) setActiveQuestion(question);
    else localStorage.removeItem(ACTIVE_QUESTION_KEY);
  };
  const toggleAnswerFrame = async row => {
    if (!activeQuestion) { setAnswerError('Chưa chọn câu hỏi để trả lời — chọn ở BỘ ĐỀ THI trước.'); return; }
    setAnswerError('');
    try {
      const response = await apiFetch(`/questions/${encodeURIComponent(activeQuestion.question_id)}/answer/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: data.item.video_id, frame_idx: row.frame_idx, pts_time: row.pts_time }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        setAnswerError(body?.detail || 'Không lưu được đáp án.');
        return;
      }
      const updated = await response.json();
      setAllQuestions(items => items.map(item => item.question_id === updated.question_id ? updated : item));
    } catch {
      setAnswerError('Không lưu được đáp án.');
    }
  };
  const isFrameAnswered = row => (activeQuestion?.answer_frames || []).some(
    frame => frame.video_id === data?.item?.video_id && frame.frame_idx === row.frame_idx
  );
  const activeKeyframeIndex = useMemo(() => {
    let index = -1;
    for (let i = 0; i < keyframeMap.length; i += 1) {
      if (keyframeMap[i].pts_time <= playbackTime) index = i; else break;
    }
    return index;
  }, [keyframeMap, playbackTime]);
  useEffect(() => {
    activeRowRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [activeKeyframeIndex]);
  useEffect(() => {
    const load = () => { if (!window.YT || !iframeRef.current) return; ytPlayerRef.current = new window.YT.Player(iframeRef.current, { events: { onReady: e => e.target.playVideo(), onStateChange: e => setPlaying(e.data === window.YT.PlayerState.PLAYING) } }); };
    if (window.YT?.Player) load(); else { const script = document.createElement('script'); script.src = 'https://www.youtube.com/iframe_api'; document.body.appendChild(script); window.onYouTubeIframeAPIReady = load; }
    return () => {
      window.onYouTubeIframeAPIReady = null;
      try { ytPlayerRef.current?.destroy?.(); } catch {}
      ytPlayerRef.current = null;
    };
  }, [data]);
  const seek = seconds => { const player = ytPlayerRef.current; if (player?.getCurrentTime) player.seekTo(Math.max(0, player.getCurrentTime() + seconds), true); };
  const seekTo = time => { const player = ytPlayerRef.current; if (player?.seekTo) { player.seekTo(Math.max(0, time), true); player.playVideo?.(); } };
  const togglePlayback = () => { const player = ytPlayerRef.current; if (!player) return; playing ? player.pauseVideo() : player.playVideo(); };
  const changeSpeed = value => { ytPlayerRef.current?.setPlaybackRate?.(value); setSpeed(value); };
  const switchVideo = video => {
    const nextData = {
      item: {
        video_id: video.video_id,
        keyframe_id: null,
        frame_idx: null,
        pts_time: 0,
        score: null,
        frame_url: video.thumbnail_url || null,
        shot_id: null,
        text: video.title || '',
      },
      modality: 'video',
      watchUrl: video.watch_url,
      title: video.title || '',
    };
    currentTimeRef.current = 0;
    setPlaybackTime(0);
    setPlaying(true);
    setSpeed(1);
    setKeyframeMap([]);
    setData(nextData);
    sessionStorage.setItem('aic-player', JSON.stringify(nextData));
    const url = new URL(window.location.href);
    url.searchParams.set('data', JSON.stringify(nextData));
    window.history.replaceState(null, '', url);
  };
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
  return <main className="player-screen"><header className="player-header"><div className="player-title"><h1>Frame Player</h1><p>{item.video_id} - {modality.toUpperCase()}</p></div><div className="keyframe-summary">{item.frame_url && <img src={item.frame_url} alt="Shot preview" />}<div><span>Video</span><b>{item.video_id}</b></div><div><span>Frame</span><b>{item.frame_idx ?? '-'}</b></div><div><span>Timestamp</span><b>{timestamp.toFixed(2)}s ({formatTimestamp(timestamp)})</b></div><div><span>Score</span><b>{item.score != null ? Number(item.score).toFixed(4) : '-'}</b></div>{item.text && modality !== 'visual' && <div className="summary-text"><span>{modality === 'video' ? 'Title' : modality.toUpperCase()}</span><b>{item.text}</b></div>}</div><VideoSearch currentVideoId={item.video_id} onSelect={switchVideo} /><div className="account-control"><Avatar username={currentUsername} title={currentUsername ? `Đăng nhập: ${currentUsername}` : ''} size="lg" /><button type="button" className="logout-button" onClick={logout}>Đăng xuất</button></div><button className="close-player" onClick={closePlayer} aria-label="Dong cua so" title="Dong cua so">&times;</button></header><div className="player-layout" style={{ '--workspace-width': `${workspaceWidth}%` }}><div className="player-main"><div className="player-video"><iframe key={`${item.video_id}-${start}`} ref={iframeRef} title="Video nguon" src={embed} allow="autoplay; encrypted-media" allowFullScreen /></div><div className="player-controls"><button onClick={() => seek(-10)}>↶<small>10</small></button><button title="Lui 5 giay" onClick={() => seek(-5)}>↶<small>5</small></button><button className="play-control" onClick={togglePlayback}>{playing ? "⏸" : "▶"}</button><button title="Tien 5 giay" onClick={() => seek(5)}>↷<small>5</small></button><button title="Tien 10 giay" onClick={() => seek(10)}>↷<small>10</small></button><span className="control-divider" />{[0.5,1,1.5,2].map(value => <button className={speed === value ? "active" : ""} onClick={() => changeSpeed(value)} key={value}>x{value}</button>)}</div></div><div className="workspace-splitter" onPointerDown={resizeWorkspace} onDoubleClick={() => setWorkspaceWidth(35)} role="separator" aria-label="Resize workspace" aria-orientation="vertical" aria-valuemin="20" aria-valuemax="55" aria-valuenow={Math.round(workspaceWidth)} /><aside className="player-workspace" aria-label="Workspace">
    <div className="active-question-bar">
      <label>Đang trả lời câu
        <select value={activeQuestionId} onChange={event => changeActiveQuestion(event.target.value)}>
          <option value="">— Chưa chọn —</option>
          {allQuestions.map(question => <option key={question.question_id} value={question.question_id}>{question.part}-{question.number} · {question.qtype.toUpperCase()}</option>)}
        </select>
      </label>
      {activeQuestion?.qtype === 'trake' && <span className="active-question-progress">{(activeQuestion.answer_frames || []).length}/{activeQuestion.events?.length || 0} khung</span>}
      {activeQuestion && activeQuestion.qtype !== 'trake' && <span className="active-question-progress">{(activeQuestion.answer_frames || []).length} khung đã chọn</span>}
      {answerError && <p className="error-banner">{answerError}</p>}
    </div>
    <div className="workspace-top">
      <h3>Keyframe hiện tại</h3>
      {activeKeyframeIndex >= 0 ? <dl className="workspace-current">
        <div><dt>n</dt><dd>{keyframeMap[activeKeyframeIndex].n}</dd></div>
        <div><dt>pts_time</dt><dd>{keyframeMap[activeKeyframeIndex].pts_time.toFixed(2)}s ({formatTimestamp(keyframeMap[activeKeyframeIndex].pts_time)})</dd></div>
        <div><dt>fps</dt><dd>{keyframeMap[activeKeyframeIndex].fps}</dd></div>
        <div><dt>frame_idx</dt><dd>{keyframeMap[activeKeyframeIndex].frame_idx}</dd></div>
      </dl> : <p className="workspace-empty">{keyframeMap.length ? 'Đang chờ video phát...' : 'Không có dữ liệu map-keyframes cho video này.'}</p>}
    </div>
    <div className="workspace-bottom">
      <h3>Map keyframes ({keyframeMap.length})</h3>
      <div className="workspace-list">
        <div className="workspace-row workspace-row-head has-answer"><span>n</span><span>pts_time</span><span>frame_idx</span><span></span></div>
        {keyframeMap.map((row, index) => {
          const answered = isFrameAnswered(row);
          return <div key={row.n} ref={index === activeKeyframeIndex ? activeRowRef : null} className={(index === activeKeyframeIndex ? 'workspace-row active' : 'workspace-row') + ' has-answer'} role="button" tabIndex={0} title={`Nhảy tới ${formatTimestamp(row.pts_time)}`} onClick={() => seekTo(row.pts_time)} onKeyDown={event => (event.key === 'Enter' || event.key === ' ') && (event.preventDefault(), seekTo(row.pts_time))}>
            <span>{row.n}</span><span>{row.pts_time.toFixed(2)}s</span><span>{row.frame_idx}</span>
            <span><button
              type="button"
              className={answered ? 'answer-frame-toggle checked' : 'answer-frame-toggle'}
              aria-pressed={answered}
              title={answered ? 'Bỏ chọn đáp án này' : 'Chọn frame này làm đáp án'}
              onClick={event => { event.stopPropagation(); toggleAnswerFrame(row); }}
            /></span>
          </div>;
        })}
      </div>
    </div>
  </aside></div></main>;
}

function MainScreen() {
  const { username: currentUsername, isAdmin } = useCurrentAccount();
  const [inputs, setInputs] = useState([{ id: 1, modality: 'visual', text: '' }, { id: 2, modality: 'asr', text: '' }]);
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const [sidebarWidth, setSidebarWidth] = useState(420);
  const [visualWidth, setVisualWidth] = useState(50);
  const [config, setConfig] = useState({ topK: 100, candidateK: 200, temporalGap: 3 });
  const [sceneFilter, setSceneFilter] = useState('all');
  const [displayView, setDisplayView] = useState('grid');
  const [resultModality, setResultModality] = useState('visual');
  const [player, setPlayer] = useState(null);
  const [scopeOptions, setScopeOptions] = useState({ batches: ['all'], directories: ['all'] });
  const [batch, setBatch] = useState('all');
  const [directory, setDirectory] = useState('all');
  const [videoScope, setVideoScope] = useState(null);

  useEffect(() => { apiFetch('/search-options').then(response => response.ok ? response.json() : null).then(data => data && setScopeOptions(data)).catch(() => {}); }, []);

  useEffect(() => {
    const videoId = new URLSearchParams(window.location.search).get('video_id');
    if (!videoId) return;
    let cancelled = false;
    apiFetch(`/videos?query=${encodeURIComponent(videoId)}&limit=1`)
      .then(response => response.ok ? response.json() : [])
      .then(videos => {
        if (cancelled || !videos?.length) return;
        const video = videos[0];
        setVideoScope({ video_id: video.video_id || videoId, title: video.title || videoId });
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const total = useMemo(() => groups.reduce((sum, group) => sum + group.items.length, 0), [groups]);
  const fusedResults = useMemo(() => fuseResults(groups), [groups]);
  const visibleGroups = useMemo(() => groups.filter(group => group.modality === resultModality), [groups, resultModality]);
  const visibleTotal = resultModality === 'fusion' ? fusedResults.length : visibleGroups.reduce((sum, group) => sum + group.items.length, 0);
  const activeTitle = inputs.filter(item => item.text.trim()).map(item => item.text.trim()).join(' + ') || 'Multimodal video retrieval';

  useEffect(() => {
    const handler = event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') search(); };
    window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler);
  });

  const patchInput = (id, patch) => setInputs(items => items.map(item => item.id === id ? { ...item, ...patch } : item));
  const addInput = () => setInputs(items => [...items, { id: Date.now(), modality: 'visual', text: '' }]);
  const selectQuestionText = text => setInputs([{ id: Date.now(), modality: 'visual', text }]);

  async function openPlayer(item, modality) {
    const response = await apiFetch(`/videos/${encodeURIComponent(item.video_id)}`);
    if (!response.ok) return setError('Không tìm thấy URL video nguồn.');
    const video = await response.json();
    const start = Math.max(0, Math.floor(modality === 'asr' ? item.start_time : item.pts_time || 0));
    sessionStorage.setItem('aic-player', JSON.stringify({ item, modality, watchUrl: video.watch_url }));
    const payload = encodeURIComponent(JSON.stringify({ item, modality, watchUrl: video.watch_url }));
    window.open(`${window.location.origin}/ui/?player=1&data=${payload}`, '_blank', 'noopener,noreferrer');
  }

  async function toggleVideoScope(videoId) {
    if (!videoId) return;
    if (videoScope?.video_id === videoId) {
      setVideoScope(null);
      return;
    }
    let title = videoId;
    try {
      const response = await apiFetch(`/videos?query=${encodeURIComponent(videoId)}&limit=1`);
      if (response.ok) {
        const videos = await response.json();
        const video = videos.find(item => item.video_id === videoId) || videos[0];
        if (video?.title) title = video.title;
      }
    } catch {}
    setVideoScope({ video_id: videoId, title });
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

  async function search() {
    const active = inputs.filter(item => item.text.trim() && MODES.some(([value]) => value === item.modality));
    if (!active.length || loading) return setError('Hãy nhập ít nhất một truy vấn.');
    setLoading(true); setGroups([]); setError(''); setElapsed(0);
    const started = performance.now(); const timer = setInterval(() => setElapsed((performance.now() - started) / 1000), 100);
    try {
      const results = await Promise.all(active.map(async item => {
        const candidateParam = item.modality === 'visual' ? `&candidate_k=${config.candidateK}` : '';
        const params = new URLSearchParams({ query: item.text.trim(), top_k: String(config.topK), batch, directory });
        if (videoScope) params.set('video_id', videoScope.video_id);
        const response = await apiFetch(`/search/${item.modality}?${params.toString()}${candidateParam}`);
        if (!response.ok) throw new Error(await response.text());
        return { modality: item.modality, query: item.text.trim(), items: await response.json() };
      }));
      setGroups(results);
      if (results.length > 1) setResultModality('fusion');
      else if (results.length) setResultModality(results[0].modality);
    } catch (exception) { setError(exception.message); }
    finally { clearInterval(timer); setElapsed((performance.now() - started) / 1000); setLoading(false); }
  }

  return <div className="app-shell" style={{ '--sidebar-width': `${sidebarWidth}px` }}>
    <aside className="sidebar">
      <header className="brand"><img src="/assets/LogoAIC_TTVN_1.png"/><div><strong>Video Search AIC 2026</strong><span className="brand-tagline"><i className="tt">TT</i><i className="v">V</i><i className="n">N</i><em> - IUH - one day for all</em></span></div><div className="account-control"><Avatar username={currentUsername} title={currentUsername ? `Đăng nhập: ${currentUsername}` : ''} size="lg" /><button type="button" className="logout-button" onClick={logout}>Đăng xuất</button></div></header>
      <QuestionBank onSelectQuestion={selectQuestionText} currentUsername={currentUsername} isAdmin={isAdmin} />
      <div className="section-heading"><span>QUERY INPUTS</span><button className="add-input-button" onClick={addInput}>+ ADD INPUT</button></div>
      <div className="query-list">{inputs.map(item => <QueryInput key={item.id} item={item} canRemove={inputs.length > 1} onChange={patch => patchInput(item.id, patch)} onRemove={() => setInputs(items => items.filter(value => value.id !== item.id))}/>)}</div>
      <button className="search-button" disabled={loading} onClick={search}>{loading ? <><i className="mini-spinner"/> Searching {elapsed.toFixed(1)}s</> : <>Search <kbd>Ctrl</kbd> <kbd>Enter</kbd></>}</button>
      {error && <p className="error-banner">{error}</p>}
      <section className="config-section"><div className="section-heading"><span>DATA SCOPE</span></div><div className="config-grid"><label>Batch<select value={batch} onChange={event => setBatch(event.target.value)}>{scopeOptions.batches.map(value => <option key={value} value={value}>{value === 'all' ? 'Tất cả batch' : value}</option>)}</select></label><label>Thư mục tìm kiếm<select value={directory} onChange={event => setDirectory(event.target.value)}>{scopeOptions.directories.map(value => <option key={value} value={value}>{value === 'all' ? 'Tất cả thư mục' : value}</option>)}</select></label></div><div className="video-scope-control"><span className="scope-caption">Khóa video</span>{videoScope ? <div className="video-scope-chip"><span><b>{videoScope.video_id}</b><small>{videoScope.title}</small></span><button type="button" onClick={() => setVideoScope(null)} aria-label="Bỏ khóa video" title="Bỏ khóa video">×</button></div> : <VideoSearch onSelect={video => setVideoScope({ video_id: video.video_id, title: video.title || video.video_id })} />}</div></section>
      <section className="config-section"><div className="section-heading"><span>CONFIGURATIONS</span></div><div className="config-grid"><label>Results<select value={config.topK} onChange={event => setConfig({...config, topK: Number(event.target.value)})}><option value={20}>20</option><option value={50}>50</option><option value={100}>100</option><option value={150}>150</option><option value={200}>200</option></select></label><label>Candidate K<input type="number" min="20" max="500" value={config.candidateK} onChange={event => setConfig({...config, candidateK: Math.max(20, Math.min(500, Number(event.target.value) || 20))})}/></label><label>Time gap (s)<input type="number" min="0" max="60" step="0.5" value={config.temporalGap} onChange={event => setConfig({...config, temporalGap: Math.max(0, Math.min(60, Number(event.target.value) || 0))})}/></label></div></section>
    </aside><div className="panel-splitter sidebar-splitter" onPointerDown={event => resizePanel(event, 'sidebar')} />
    <main className="results-pane">
      <header className="results-header"><div><span className="results-label">RESULTS <b>{loading ? `${elapsed.toFixed(1)}s` : visibleTotal}</b></span><h1>{activeTitle}</h1><div className="result-summary"><small>WHOLE</small>{videoScope && <span className="video-scope-badge">VIDEO {videoScope.video_id}</span>}</div></div><div className="result-meta"><div className="result-type-switcher">{[['fusion','Fusion'],['visual','Visual'],['ocr','OCR'],['asr','ASR'],['object','Object'],['topic','Topic']].map(([value,label]) => <button type="button" key={value} disabled={value === 'fusion' && groups.length < 2} className={resultModality === value ? 'active' : ''} onClick={() => setResultModality(value)}>{label}</button>)}</div><select className="view-select" aria-label="Result view" value={displayView} onChange={event => setDisplayView(event.target.value)}><option value="grid">Grid View</option><option value="cluster">Cluster View</option></select><select aria-label="Filter scene type" value={sceneFilter} onChange={event => setSceneFilter(event.target.value)}><option value="all">All scenes</option><option value="daytime">Daytime</option><option value="night">Night</option><option value="rain">Rain</option><option value="sunny">Sunny</option><option value="indoor">Indoor</option><option value="outdoor">Outdoor</option></select></div></header>
      {loading && <div className="loading-state"><i className="spinner"/><strong>Đang tìm kiếm video</strong><span>Querying indexes and ranking candidates...</span><time>{elapsed.toFixed(1)}s</time></div>}
      {!loading && !groups.length && <div className="welcome"><strong>Bắt đầu bằng một truy vấn</strong><p>Nhập mô tả hình ảnh hoặc lời thoại tiếng Việt ở cột bên trái.</p></div>}
      {!loading && resultModality === 'fusion' && groups.length > 0 && <div className={`result-columns view-${displayView}`}><section className="result-group" style={{width: '100%'}}><header><h2>Fusion (Reciprocal Rank Fusion)</h2><span>{fusedResults.length}</span></header><p className="group-query">Kết hợp {groups.length} input: {groups.map(group => `${group.modality}:"${group.query}"`).join(', ')}</p><div className="result-grid">{fusedResults.length ? fusedResults.map((entry,index) => <FusionCard key={entry.key} entry={entry} index={index} onOpen={openPlayer}/>) : <div className="empty-state">Không tìm thấy kết quả phù hợp.</div>}</div></section></div>}
      {!loading && resultModality !== 'fusion' && groups.length > 0 && visibleGroups.length === 0 && <div className="empty-state">No {resultModality.toUpperCase()} results available.</div>}
      {!loading && resultModality !== 'fusion' && visibleGroups.length > 0 && <div className={`result-columns view-${displayView}`}>{visibleGroups.map((group,index) => <><section className="result-group" style={{width: '100%'}} key={`${group.modality}-${group.query}`}><header><h2>{{visual:'Visual / Text',asr:'ASR transcript',ocr:'OCR text',object:'Object detection',topic:'Video Topic'}[group.modality]}</h2><span>{group.items.length}</span></header><p className="group-query">“{group.query}”</p>{!group.items.length ? <div className="empty-state">Không tìm thấy kết quả phù hợp.</div> : displayView === 'cluster' ? <div className="result-clusters">{groupByVideo(group.items).map(cluster => { const videoId = cluster.items[0]?.video_id || cluster.key; const locked = videoScope?.video_id === videoId; return <div className="result-cluster-row" key={cluster.key}><div className="result-cluster-video"><button type="button" className={`video-lock-button${locked ? ' active' : ''}`} onClick={() => toggleVideoScope(videoId)} aria-label={locked ? `Bỏ khóa ${videoId}` : `Khóa tìm kiếm vào ${videoId}`} aria-pressed={locked} title={locked ? 'Bỏ khóa video' : 'Khóa tìm kiếm vào video này'}><LockIcon locked={locked}/></button><strong>{videoId}</strong>{locked && <small>Đang khóa</small>}</div><div className="result-cluster-items">{cluster.items.map((item,itemIndex) => <ResultCard key={item.keyframe_id || item.segment_id || item.video_id} item={item} index={itemIndex} modality={group.modality} onOpen={openPlayer}/>)}</div></div>; })}</div> : <div className="result-grid">{group.items.map((item,itemIndex) => <ResultCard key={item.keyframe_id || item.segment_id || item.video_id} item={item} index={itemIndex} modality={group.modality} onOpen={openPlayer}/>)}</div>}</section></>)}</div>}
      {player && <div className="player-backdrop" onMouseDown={event => event.target === event.currentTarget && setPlayer(null)}><section className="player-modal"><header><div><strong>{player.video_id}</strong><span>{player.start}s</span></div><button aria-label="Close video" onClick={() => setPlayer(null)}>×</button></header>{player.embedUrl ? <iframe src={player.embedUrl} title={player.video_id} allow="autoplay; encrypted-media; picture-in-picture" allowFullScreen/> : <img src={player.frame_url}/>}<footer><a href={player.watchUrl} target="_blank" rel="noreferrer">Open video at {player.start}s</a></footer></section></div>}
    </main>
  </div>;
}

function AuthGate({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem(AUTH_STORAGE_KEY));
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const onExpired = () => setToken(null);
    window.addEventListener('aic-auth-expired', onExpired);
    return () => window.removeEventListener('aic-auth-expired', onExpired);
  }, []);

  const submit = async event => {
    event.preventDefault();
    if (!username.trim() || !password) return;
    setSubmitting(true);
    setError('');
    try {
      const response = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      if (!response.ok) throw new Error('Sai tài khoản hoặc mật khẩu.');
      const data = await response.json();
      localStorage.setItem(AUTH_STORAGE_KEY, data.token);
      // Reload so every already-mounted component (map-keyframes, search-options,
      // question bank, ...) refetches with the token now present, instead of staying
      // stuck with whatever failed before login.
      window.location.reload();
      return;
    } catch (exception) {
      setError(exception.message || 'Đăng nhập thất bại.');
    } finally {
      setSubmitting(false);
    }
  };

  return <>
    <div className={token ? 'auth-guarded' : 'auth-guarded auth-blurred'} aria-hidden={!token}>{children}</div>
    {!token && <div className="auth-overlay">
      <form className="auth-card" onSubmit={submit}>
        <img src="/assets/LogoAIC_TTVN_1.png" alt="" />
        <h1>Đăng nhập</h1>
        <label>Tài khoản<input value={username} onChange={event => setUsername(event.target.value)} autoFocus autoComplete="username" /></label>
        <label>Mật khẩu<input type="password" value={password} onChange={event => setPassword(event.target.value)} autoComplete="current-password" /></label>
        {error && <p className="auth-error">{error}</p>}
        <button type="submit" disabled={submitting || !username.trim() || !password}>{submitting ? 'Đang đăng nhập...' : 'Đăng nhập'}</button>
      </form>
    </div>}
  </>;
}

export default function App() {
  const isPlayer = new URLSearchParams(window.location.search).get('player') === '1';
  return <AuthGate>{isPlayer ? <PlayerScreen /> : <MainScreen />}</AuthGate>;
}

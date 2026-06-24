import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { 
  getServices, 
  startService, 
  stopService, 
  restartService, 
  getServiceLogs, 
  updateServiceConfig 
} from '../services/api';

const ServiceControlDashboard = () => {
  const [services, setServices] = useState([]);
  const [selectedService, setSelectedService] = useState(null);
  const [logs, setLogs] = useState([]);
  const [logFilter, setLogFilter] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);
  const [configVars, setConfigVars] = useState({ key: '', value: '' });
  const [loadingAction, setLoadingAction] = useState({});

  const terminalEndRef = useRef(null);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [historyData, setHistoryData] = useState({}); // Stores resource history for chart: { service_name: { times: [], cpu: [], mem: [] } }

  // 1. WebSocket / Polling Connection for Status
  useEffect(() => {
    let socket = null;
    let pollInterval = null;

    const handleStatusUpdate = (data) => {
      setServices(data);
      
      // Update history for charts
      const now = new Date().toLocaleTimeString();
      setHistoryData(prev => {
        const next = { ...prev };
        data.forEach(s => {
          if (!next[s.name]) {
            next[s.name] = { times: [], cpu: [], mem: [] };
          }
          const sHistory = next[s.name];
          sHistory.times.push(now);
          sHistory.cpu.push(s.cpu_percent || 0);
          sHistory.mem.push(s.memory_mb || 0);
          
          // Keep only last 20 points
          if (sHistory.times.length > 20) {
            sHistory.times.shift();
            sHistory.cpu.shift();
            sHistory.mem.shift();
          }
        });
        return next;
      });

      // Automatically select the first service if none is selected
      if (data.length > 0 && !selectedService) {
        setSelectedService(data[0].name);
      }
    };

    // Try WebSocket connection first
    try {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ws/services/status`;
      console.log('[Orchestrator WS] Connecting to:', wsUrl);
      socket = new WebSocket(wsUrl);

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'services_status') {
          handleStatusUpdate(msg.data);
        }
      };

      socket.onerror = (err) => {
        console.warn('[Orchestrator WS] Error, falling back to HTTP polling:', err);
        startPolling();
      };

      socket.onclose = () => {
        console.warn('[Orchestrator WS] Closed, falling back to HTTP polling');
        startPolling();
      };
    } catch (e) {
      console.error('[Orchestrator WS] Setup error, falling back to HTTP polling:', e);
      startPolling();
    }

    const startPolling = () => {
      const fetchStatus = async () => {
        try {
          const data = await getServices();
          handleStatusUpdate(data);
        } catch (err) {
          console.error('Error polling services:', err);
        }
      };
      fetchStatus();
      pollInterval = setInterval(fetchStatus, 2000);
    };

    return () => {
      if (socket) socket.close();
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [selectedService]);

  // 2. Log Polling
  useEffect(() => {
    if (!selectedService || isPaused) return;

    const fetchLogs = async () => {
      try {
        const data = await getServiceLogs(selectedService, 120);
        setLogs(data.logs || []);
      } catch (err) {
        console.error('Error fetching logs:', err);
      }
    };

    fetchLogs();
    const logInterval = setInterval(fetchLogs, 2000);

    return () => clearInterval(logInterval);
  }, [selectedService, isPaused]);

  // 3. Auto-scroll Terminal
  useEffect(() => {
    if (autoScroll && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // 4. ECharts Resource Visualizer
  useEffect(() => {
    if (!chartRef.current || !selectedService) return;

    const history = historyData[selectedService] || { times: [], cpu: [], mem: [] };

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const option = {
      title: {
        text: 'Resource Usage (Last 30s)',
        textStyle: { color: '#a0aec0', fontSize: 14, fontWeight: 'normal' },
        left: 'center'
      },
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['CPU %', 'Memory (MB)'],
        textStyle: { color: '#718096' },
        bottom: 0
      },
      grid: { left: '4%', right: '4%', top: '15%', bottom: '18%', containLabel: true },
      xAxis: {
        type: 'category',
        data: history.times,
        axisLine: { lineStyle: { color: '#4a5568' } },
        axisLabel: { color: '#718096' }
      },
      yAxis: [
        {
          type: 'value',
          name: 'CPU %',
          min: 0,
          max: 100,
          axisLine: { lineStyle: { color: '#48bb78' } },
          axisLabel: { color: '#718096' },
          splitLine: { lineStyle: { color: '#2d3748' } }
        },
        {
          type: 'value',
          name: 'RAM (MB)',
          axisLine: { lineStyle: { color: '#4299e1' } },
          axisLabel: { color: '#718096' },
          splitLine: { show: false }
        }
      ],
      series: [
        {
          name: 'CPU %',
          type: 'line',
          data: history.cpu,
          smooth: true,
          showSymbol: false,
          itemStyle: { color: '#48bb78' },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(72,187,120,0.4)' },
              { offset: 1, color: 'rgba(72,187,120,0.0)' }
            ])
          }
        },
        {
          name: 'Memory (MB)',
          type: 'line',
          yAxisIndex: 1,
          data: history.mem,
          smooth: true,
          showSymbol: false,
          itemStyle: { color: '#4299e1' },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(66,153,225,0.4)' },
              { offset: 1, color: 'rgba(66,153,225,0.0)' }
            ])
          }
        }
      ],
      backgroundColor: 'transparent'
    };

    chartInstance.current.setOption(option);

    const handleResize = () => {
      if (chartInstance.current) chartInstance.current.resize();
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [selectedService, historyData]);

  // 5. Actions: Start, Stop, Restart
  const handleServiceAction = async (name, action) => {
    setLoadingAction(prev => ({ ...prev, [`${name}-${action}`]: true }));
    try {
      let res;
      if (action === 'start') res = await startService(name);
      else if (action === 'stop') res = await stopService(name);
      else if (action === 'restart') res = await restartService(name);
      
      console.log(`Action ${action} on ${name} completed:`, res);
      
      // Instantly update status locally to improve responsiveness
      setServices(prev => 
        prev.map(s => {
          if (s.name === name) {
            return { 
              ...s, 
              status: action === 'start' ? 'RUNNING' : action === 'stop' ? 'STOPPED' : 'RUNNING',
              uptime_seconds: action === 'start' ? 0 : s.uptime_seconds
            };
          }
          return s;
        })
      );
    } catch (err) {
      console.error(`Failed to perform ${action} on ${name}:`, err);
      alert(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoadingAction(prev => ({ ...prev, [`${name}-${action}`]: false }));
    }
  };

  // Helper for styling logs based on content
  const formatLogLine = (line) => {
    if (line.includes('INFO') || line.includes('info:')) {
      return <span className="text-green-400">{line}</span>;
    } else if (line.includes('WARNING') || line.includes('warn:')) {
      return <span className="text-yellow-400 font-medium">{line}</span>;
    } else if (line.includes('ERROR') || line.includes('CRITICAL') || line.includes('fail:')) {
      return <span className="text-red-400 font-semibold">{line}</span>;
    } else if (line.includes('[System]') || line.includes('[Orchestrator')) {
      return <span className="text-cyan-400 italic">{line}</span>;
    } else if (line.includes('TRADE EXECUTED')) {
      return <span className="text-emerald-300 font-bold bg-emerald-950/40 px-1 rounded">{line}</span>;
    }
    return <span className="text-gray-300">{line}</span>;
  };

  const filteredLogs = logs.filter(line => 
    line.toLowerCase().includes(logFilter.toLowerCase())
  );

  const getStatusBadge = (status, lastError) => {
    switch (status) {
      case 'RUNNING':
        return (
          <div className="flex items-center gap-2 text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20 text-xs font-semibold">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            RUNNING
          </div>
        );
      case 'STOPPED':
        return (
          <div className="flex items-center gap-2 text-gray-400 bg-gray-500/10 px-3 py-1 rounded-full border border-gray-500/20 text-xs font-semibold">
            <span className="h-2 w-2 rounded-full bg-gray-500" />
            STOPPED
          </div>
        );
      case 'ERROR':
        return (
          <div 
            title={lastError}
            className="flex items-center gap-2 text-rose-400 bg-rose-500/10 px-3 py-1 rounded-full border border-rose-500/20 text-xs font-semibold cursor-help"
          >
            <span className="h-2 w-2 rounded-full bg-rose-400 animate-ping" />
            ERROR
          </div>
        );
      case 'FINISHED':
        return (
          <div className="flex items-center gap-2 text-sky-400 bg-sky-500/10 px-3 py-1 rounded-full border border-sky-500/20 text-xs font-semibold">
            <span className="h-2 w-2 rounded-full bg-sky-400" />
            FINISHED
          </div>
        );
      default:
        return (
          <div className="flex items-center gap-2 text-yellow-400 bg-yellow-500/10 px-3 py-1 rounded-full border border-yellow-500/20 text-xs font-semibold">
            <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
            PENDING
          </div>
        );
    }
  };

  const activeServiceObj = services.find(s => s.name === selectedService);

  return (
    <div className="flex flex-col gap-6 text-gray-100 min-h-[700px] bg-slate-950 p-6 rounded-2xl border border-slate-800 shadow-2xl relative overflow-hidden backdrop-blur-md">
      {/* Background radial glow */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl -z-10 pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl -z-10 pointer-events-none" />

      {/* Header */}
      <div className="flex justify-between items-center border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent flex items-center gap-3">
            <svg className="w-7 h-7 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            Service Orchestrator
          </h2>
          <p className="text-sm text-slate-400 mt-1">Control and monitor all decentralized Python microservices from a single container-independent interface.</p>
        </div>
        <div className="text-xs text-slate-500 bg-slate-900/80 px-3 py-1.5 rounded-md border border-slate-800">
          Environment: <span className="text-indigo-400 font-semibold">Local Subprocess (Host-Level)</span>
        </div>
      </div>

      {/* Main Container Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-stretch">
        
        {/* Left Side: Service Grid List (7 Cols on large screens) */}
        <div className="xl:col-span-7 flex flex-col gap-4">
          <h3 className="text-base font-semibold text-slate-300 flex items-center gap-2">
            Microservices Directory
            <span className="text-xs bg-slate-800 px-2 py-0.5 rounded-full text-slate-400">{services.length} Total</span>
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[560px] overflow-y-auto pr-2 custom-scrollbar">
            {services.map((s) => {
              const isSelected = s.name === selectedService;
              return (
                <div 
                  key={s.name}
                  onClick={() => setSelectedService(s.name)}
                  className={`flex flex-col gap-3 p-4 rounded-xl cursor-pointer transition-all duration-200 relative overflow-hidden ${
                    isSelected 
                      ? 'bg-slate-900/90 border border-indigo-500/50 shadow-indigo-500/5 shadow-md scale-[1.01]' 
                      : 'bg-slate-900/40 hover:bg-slate-900/60 border border-slate-800 hover:border-slate-700'
                  }`}
                >
                  {/* Card Glow if active */}
                  {isSelected && (
                    <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/10 rounded-full blur-2xl -z-10 pointer-events-none" />
                  )}

                  {/* Title & Status */}
                  <div className="flex justify-between items-start gap-2">
                    <div>
                      <h4 className="font-semibold text-white tracking-wide">{s.display_name}</h4>
                      <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider bg-slate-950 px-2 py-0.5 rounded border border-slate-900 mt-1 inline-block">
                        {s.name}
                      </span>
                    </div>
                    {getStatusBadge(s.status, s.last_error)}
                  </div>

                  {/* Description */}
                  <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed h-8">
                    {s.description}
                  </p>

                  {/* Metrics Row */}
                  {s.status === 'RUNNING' ? (
                    <div className="grid grid-cols-3 gap-2 py-1.5 bg-slate-950/60 rounded-lg px-3 border border-slate-900 text-[11px] font-mono">
                      <div className="flex flex-col">
                        <span className="text-[10px] text-slate-500">CPU</span>
                        <span className="text-emerald-400 font-semibold">{s.cpu_percent}%</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[10px] text-slate-500">MEMORY</span>
                        <span className="text-blue-400 font-semibold">{s.memory_mb} MB</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[10px] text-slate-500">UPTIME</span>
                        <span className="text-indigo-400 font-semibold">
                          {s.uptime_seconds > 60 
                            ? `${Math.floor(s.uptime_seconds / 60)}m ${Math.floor(s.uptime_seconds % 60)}s` 
                            : `${Math.floor(s.uptime_seconds)}s`
                          }
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center py-1.5 bg-slate-950/20 rounded-lg text-[11px] font-mono border border-slate-900/50 text-slate-500">
                      Inactive • Resources Cleared
                    </div>
                  )}

                  {/* Actions buttons inside card */}
                  <div className="flex gap-2 border-t border-slate-900 pt-3 mt-1 justify-end items-center" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => setSelectedService(s.name)}
                      title="Inspect Logs"
                      className="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1 transition-colors border border-slate-700/50"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      Logs
                    </button>

                    <button
                      onClick={() => setEditingConfig(s)}
                      title="Configure Environment"
                      className="text-xs p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border border-slate-700/50"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </button>

                    {s.name !== 'serve' && (
                      <>
                        {s.status === 'RUNNING' ? (
                          <>
                            <button
                              onClick={() => handleServiceAction(s.name, 'stop')}
                              disabled={loadingAction[`${s.name}-stop`]}
                              className="text-xs px-2.5 py-1 rounded bg-rose-950/50 hover:bg-rose-900 text-rose-300 border border-rose-800/40 font-semibold transition-colors flex items-center gap-1 disabled:opacity-55"
                            >
                              <span className="w-2 h-2 rounded-sm bg-rose-400" />
                              Stop
                            </button>
                            <button
                              onClick={() => handleServiceAction(s.name, 'restart')}
                              disabled={loadingAction[`${s.name}-restart`]}
                              className="text-xs p-1 rounded bg-indigo-950/30 hover:bg-indigo-900/50 text-indigo-300 border border-indigo-800/20 transition-colors disabled:opacity-55"
                              title="Restart Service"
                            >
                              <svg className={`w-3.5 h-3.5 ${loadingAction[`${s.name}-restart`] ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 8H18.5" />
                              </svg>
                            </button>
                          </>
                        ) : (
                          <button
                            onClick={() => handleServiceAction(s.name, 'start')}
                            disabled={loadingAction[`${s.name}-start`]}
                            className="text-xs px-2.5 py-1 rounded bg-emerald-950/50 hover:bg-emerald-900 text-emerald-300 border border-emerald-800/40 font-semibold transition-colors flex items-center gap-1 disabled:opacity-55"
                          >
                            <svg className="w-2.5 h-2.5 fill-current" viewBox="0 0 24 24">
                              <path d="M8 5v14l11-7z" />
                            </svg>
                            Start
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Side: Active Service Resource Stats & Terminal Logs (5 Cols) */}
        <div className="xl:col-span-5 flex flex-col gap-4">
          {activeServiceObj ? (
            <>
              <div className="flex justify-between items-center border-b border-slate-900 pb-2">
                <div>
                  <h3 className="text-base font-semibold text-slate-200">{activeServiceObj.display_name}</h3>
                  <p className="text-xs text-slate-500 font-mono">Process ID: {activeServiceObj.pid || 'N/A'}</p>
                </div>
                {getStatusBadge(activeServiceObj.status, activeServiceObj.last_error)}
              </div>

              {/* Resource Mini Chart (ECharts wrapper) */}
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 h-48 flex items-center justify-center">
                {activeServiceObj.status === 'RUNNING' ? (
                  <div ref={chartRef} className="w-full h-full" />
                ) : (
                  <div className="text-center p-4">
                    <svg className="w-8 h-8 text-slate-600 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
                    </svg>
                    <p className="text-xs text-slate-500">Service is inactive. Start the service to visualize real-time CPU & Memory utilization charts.</p>
                  </div>
                )}
              </div>

              {/* Terminal Logs Panel */}
              <div className="flex flex-col gap-2 flex-grow">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-400 flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
                    Terminal Output Logs
                  </span>
                  
                  <div className="flex items-center gap-2">
                    {/* Log Filter Input */}
                    <input
                      type="text"
                      placeholder="Filter keyword..."
                      value={logFilter}
                      onChange={(e) => setLogFilter(e.target.value)}
                      className="bg-slate-950 border border-slate-800 rounded px-2 py-0.5 text-[10px] text-slate-300 focus:outline-none focus:border-slate-700 w-28"
                    />

                    {/* Pause Button */}
                    <button
                      onClick={() => setIsPaused(!isPaused)}
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold border transition-all ${
                        isPaused 
                          ? 'bg-amber-950/40 text-amber-300 border-amber-800/50 hover:bg-amber-900/50' 
                          : 'bg-slate-850 text-slate-300 border-slate-700 hover:bg-slate-750'
                      }`}
                    >
                      {isPaused ? 'Resuming...' : 'Pause Stream'}
                    </button>

                    {/* Auto Scroll Checkbox */}
                    <label className="flex items-center gap-1 text-[10px] text-slate-500 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={autoScroll}
                        onChange={(e) => setAutoScroll(e.target.checked)}
                        className="rounded bg-slate-950 border-slate-800 text-indigo-600 focus:ring-0 focus:ring-offset-0 w-3 h-3"
                      />
                      AutoScroll
                    </label>
                  </div>
                </div>

                {/* Simulated ANSI Terminal */}
                <div className="bg-slate-950 rounded-xl border border-slate-800 p-4 font-mono text-[11px] leading-5 h-80 overflow-y-auto custom-scrollbar flex flex-col gap-1 shadow-inner relative">
                  {filteredLogs.length > 0 ? (
                    filteredLogs.map((line, i) => (
                      <div key={i} className="whitespace-pre-wrap break-all border-b border-slate-950 py-0.5">
                        {formatLogLine(line)}
                      </div>
                    ))
                  ) : (
                    <div className="text-slate-600 text-center my-auto">
                      {logFilter ? 'No logs match your filter query.' : 'Waiting for incoming logs stream...'}
                    </div>
                  )}
                  <div ref={terminalEndRef} />
                </div>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-slate-500 py-12">
              <svg className="w-12 h-12 text-slate-700 mb-3 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <p className="text-sm">Select a microservice from the directory to monitor resource utilization charts and inspect streaming logs.</p>
            </div>
          )}
        </div>
      </div>

      {/* 6. Configuration Drawer Overlay */}
      {editingConfig && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex justify-end transition-opacity duration-300">
          <div className="bg-slate-900 border-l border-slate-800 w-full max-w-md h-full p-6 shadow-2xl flex flex-col gap-6 relative animate-slide-in">
            
            <div className="flex justify-between items-center border-b border-slate-800 pb-4">
              <div>
                <h3 className="text-lg font-bold text-white">Configure Environment</h3>
                <p className="text-xs text-indigo-400 font-mono">{editingConfig.display_name}</p>
              </div>
              <button 
                onClick={() => setEditingConfig(null)}
                className="text-slate-500 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex flex-col gap-4 flex-grow overflow-y-auto pr-2 custom-scrollbar">
              <p className="text-xs text-slate-400 leading-relaxed">
                Add service-specific environment variables. These parameters will be injected into the subprocess runtime context.
              </p>

              {/* Display existing config overrides */}
              <div className="flex flex-col gap-2">
                <h4 className="text-xs font-semibold text-slate-300">Active Parameters Override</h4>
                
                {Object.keys(editingConfig.config?.env_overrides || {}).length > 0 ? (
                  <div className="flex flex-col gap-1.5 font-mono text-xs bg-slate-950 rounded-lg p-3 border border-slate-800">
                    {Object.entries(editingConfig.config.env_overrides).map(([k, v]) => (
                      <div key={k} className="flex justify-between items-center gap-2 py-0.5 border-b border-slate-900 last:border-0">
                        <span className="text-slate-400">{k}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-indigo-300 font-semibold">{v}</span>
                          <button
                            onClick={() => {
                              const updated = { ...editingConfig.config.env_overrides };
                              delete updated[k];
                              updateServiceConfig(editingConfig.name, updated).then(() => {
                                editingConfig.config.env_overrides = updated;
                                setServices([...services]);
                              });
                            }}
                            className="text-rose-500 hover:text-rose-400"
                            title="Delete parameter"
                          >
                            ×
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center p-4 bg-slate-950/40 border border-dashed border-slate-800 rounded-lg text-slate-600 text-xs">
                    No custom overrides registered. Subprocess will inherit default global configurations.
                  </div>
                )}
              </div>

              {/* Add Parameter Form */}
              <div className="flex flex-col gap-3 bg-slate-950 p-4 rounded-xl border border-slate-800 mt-2">
                <h4 className="text-xs font-semibold text-slate-300">Inject Parameter</h4>
                
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-slate-500 uppercase font-mono">Key</label>
                    <input
                      type="text"
                      placeholder="e.g. MIN_VALID_FEEDS"
                      value={configVars.key}
                      onChange={(e) => setConfigVars({ ...configVars, key: e.target.value.toUpperCase() })}
                      className="bg-slate-900 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-slate-700 font-mono"
                    />
                  </div>
                  <div className="flex flex-col gap-1">
                    <label className="text-[10px] text-slate-500 uppercase font-mono">Value</label>
                    <input
                      type="text"
                      placeholder="e.g. 5"
                      value={configVars.value}
                      onChange={(e) => setConfigVars({ ...configVars, value: e.target.value })}
                      className="bg-slate-900 border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-slate-700 font-mono"
                    />
                  </div>
                </div>

                <button
                  onClick={async () => {
                    if (!configVars.key || !configVars.value) return;
                    const newOverrides = { 
                      ...(editingConfig.config?.env_overrides || {}),
                      [configVars.key]: configVars.value 
                    };
                    try {
                      await updateServiceConfig(editingConfig.name, newOverrides);
                      editingConfig.config.env_overrides = newOverrides;
                      setServices([...services]);
                      setConfigVars({ key: '', value: '' });
                    } catch (err) {
                      alert('Failed to save configuration');
                    }
                  }}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 text-white py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-colors shadow-lg shadow-indigo-500/10 border border-indigo-500/20"
                >
                  Save Override
                </button>
              </div>

            </div>

            {/* Footer Alert */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-[11px] text-slate-500 leading-relaxed mt-auto">
              <span className="text-amber-400 font-semibold uppercase block mb-1">💡 Notice</span>
              Modifying active process environments requires cycling the subprocess. Please **Restart** the service from the dashboard to apply new configurations.
            </div>

          </div>
        </div>
      )}

    </div>
  );
};

export default ServiceControlDashboard;

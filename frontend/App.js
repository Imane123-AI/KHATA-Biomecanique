import { useState, useEffect, useRef } from "react";
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend
} from "recharts";

// ── Palette ──────────────────────────────────────────────────────────────────
const C = {
  bg:      "#080C14",
  surface: "#0D1220",
  card:    "#111827",
  glass:   "rgba(255,255,255,0.04)",
  border:  "#1E2D45",
  accent:  "#3B82F6",
  teal:    "#14B8A6",
  green:   "#22C55E",
  amber:   "#F59E0B",
  red:     "#EF4444",
  purple:  "#8B5CF6",
  pri:     "#F1F5F9",
  sec:     "#94A3B8",
  mut:     "#475569",
};

// ── CSS injected once ─────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;700&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #080C14; color: #F1F5F9; font-family: 'Inter', sans-serif; }
  ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #080C14; }
  ::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 3px; }

  @keyframes fadeUp   { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:translateY(0); } }
  @keyframes fadeIn   { from { opacity:0; } to { opacity:1; } }
  @keyframes pulse    { 0%,100% { opacity:1; } 50% { opacity:.5; } }
  @keyframes float    { 0%,100% { transform:translateY(0px); } 50% { transform:translateY(-10px); } }
  @keyframes spin     { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
  @keyframes scanline { 0% { top:-10%; } 100% { top:110%; } }
  @keyframes glow     { 0%,100% { box-shadow:0 0 20px rgba(59,130,246,.3); } 50% { box-shadow:0 0 40px rgba(59,130,246,.6); } }
  @keyframes shimmer  { 0% { background-position:-200% center; } 100% { background-position:200% center; } }
  @keyframes ripple   { 0% { transform:scale(.8); opacity:1; } 100% { transform:scale(2.4); opacity:0; } }

  .fade-up  { animation: fadeUp .6s cubic-bezier(.16,1,.3,1) both; }
  .fade-in  { animation: fadeIn .4s ease both; }
  .float    { animation: float 3s ease-in-out infinite; }
  .glow-btn { animation: glow 2s ease-in-out infinite; }

  .glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
  }
  .hover-card { transition: transform .2s, border-color .2s, box-shadow .2s; cursor:pointer; }
  .hover-card:hover { transform:translateY(-3px); border-color:rgba(59,130,246,.4) !important; box-shadow:0 8px 32px rgba(59,130,246,.15); }

  .nav-link { transition: color .15s, background .15s; }
  .nav-link:hover { color:#F1F5F9 !important; background:rgba(255,255,255,.06) !important; }

  .input-field {
    width:100%; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
    border-radius:10px; padding:13px 16px 13px 44px; color:#F1F5F9; font-size:14px;
    outline:none; transition:border-color .2s, box-shadow .2s; font-family:'Inter',sans-serif;
  }
  .input-field:focus { border-color:rgba(59,130,246,.6); box-shadow:0 0 0 3px rgba(59,130,246,.15); }
  .input-field::placeholder { color:#475569; }

  .login-btn {
    width:100%; background:linear-gradient(135deg,#3B82F6,#6366F1); border:none;
    border-radius:10px; padding:14px; color:#fff; font-size:15px; font-weight:700;
    cursor:pointer; transition:transform .15s, box-shadow .15s; font-family:'Inter',sans-serif;
    letter-spacing:.3px;
  }
  .login-btn:hover { transform:translateY(-1px); box-shadow:0 8px 24px rgba(59,130,246,.4); }
  .login-btn:active { transform:translateY(0); }

  .role-btn {
    flex:1; padding:10px 8px; border-radius:8px; font-size:12px; font-weight:600;
    cursor:pointer; transition:all .15s; border:1px solid rgba(255,255,255,.1);
    background:transparent; font-family:'Inter',sans-serif;
  }
  .role-btn.active { background:rgba(59,130,246,.2); border-color:rgba(59,130,246,.5); color:#3B82F6; }
  .role-btn:not(.active) { color:#475569; }
  .role-btn:not(.active):hover { color:#94A3B8; border-color:rgba(255,255,255,.2); }
`;

// ── Mock data ─────────────────────────────────────────────────────────────────
const PARTICIPANTS = [
  { id:"P1", r_moy:30.62, score:38.9, z:0.046,  niveau:"attention", conditions:28 },
  { id:"P2", r_moy:30.79, score:54.2, z:0.227,  niveau:"attention", conditions:28 },
  { id:"P3", r_moy:30.58, score:36.9, z:0.009,  niveau:"ecart",     conditions:27 },
  { id:"P4", r_moy:30.46, score:28.2, z:-0.125, niveau:"ecart",     conditions:27 },
  { id:"P5", r_moy:30.55, score:34.1, z:-0.033, niveau:"ecart",     conditions:28 },
  { id:"P6", r_moy:30.38, score:23.6, z:-0.233, niveau:"ecart",     conditions:31 },
  { id:"P7", r_moy:30.70, score:45.3, z:0.135,  niveau:"attention", conditions:27 },
  { id:"P8", r_moy:30.67, score:43.5, z:0.101,  niveau:"attention", conditions:28 },
  { id:"P9", r_moy:30.48, score:29.6, z:-0.109, niveau:"ecart",     conditions:27 },
];
const VITESSE_DATA = [
  {v:"0.6",r:29.27},{v:"0.7",r:29.65},{v:"0.8",r:29.98},{v:"0.9",r:30.28},
  {v:"1.0",r:30.51},{v:"1.1",r:30.68},{v:"1.2",r:30.79},{v:"1.3",r:30.74},
  {v:"1.4",r:30.62},{v:"1.5",r:30.41},{v:"1.6",r:30.12},{v:"1.7",r:29.78},{v:"1.8",r:29.40},
];
const ARTICULAIRE_DATA = [
  {name:"Hanche",pct:32.5,color:C.accent},{name:"Genou",pct:27.5,color:C.teal},
  {name:"Cheville",pct:23.6,color:C.green},{name:"GRF",pct:16.4,color:C.purple},
];
const IMPORTANCE_DATA = [
  {name:"Couples hanche",pct:17.7,color:C.accent},{name:"GRF (forces sol)",pct:16.4,color:C.teal},
  {name:"Couples cheville",pct:15.2,color:C.green},{name:"Couples genou",pct:14.8,color:C.purple},
  {name:"Angles hanche",pct:11.7,color:"#FB923C"},{name:"Angles genou",pct:9.8,color:"#F472B6"},
  {name:"Angles cheville",pct:6.3,color:C.amber},{name:"Vitesses",pct:8.1,color:C.mut},
];
const DIST_REF = Array.from({length:40},(_,i)=>{
  const x=28.5+i*0.075, mu=30.58, sigma=1.01;
  return {x:x.toFixed(2), y:parseFloat((Math.exp(-0.5*((x-mu)/sigma)**2)/(sigma*Math.sqrt(2*Math.PI))).toFixed(4))};
});

// ── Shared UI ─────────────────────────────────────────────────────────────────
const Badge = ({niveau}) => {
  const cfg = {
    normal:    {label:"Normale",  bg:"#064E3B",text:C.green,  icon:"✓"},
    attention: {label:"Attention",bg:"#78350F",text:C.amber,  icon:"⚠"},
    ecart:     {label:"Ecart",    bg:"#7F1D1D",text:C.red,    icon:"✕"},
  }[niveau]||{label:niveau,bg:C.border,text:C.sec,icon:"•"};
  return (
    <span style={{background:cfg.bg,color:cfg.text,padding:"3px 10px",borderRadius:99,
      fontSize:11,fontWeight:700,letterSpacing:.3,display:"inline-flex",alignItems:"center",gap:4}}>
      {cfg.icon} {cfg.label}
    </span>
  );
};

const StatCard = ({label,value,sub,accent}) => (
  <div className="hover-card" style={{background:C.card,border:`1px solid ${C.border}`,
    borderRadius:12,padding:"20px 22px",borderLeft:`3px solid ${accent||C.accent}`}}>
    <div style={{color:C.sec,fontSize:11,fontWeight:700,letterSpacing:1.2,
      textTransform:"uppercase",marginBottom:6}}>{label}</div>
    <div style={{color:C.pri,fontSize:26,fontWeight:800,fontFamily:"'Space Grotesk',sans-serif",lineHeight:1}}>{value}</div>
    {sub&&<div style={{color:C.mut,fontSize:11,marginTop:6}}>{sub}</div>}
  </div>
);

const SectionTitle = ({children}) => (
  <h2 style={{color:C.pri,fontSize:17,fontWeight:700,marginBottom:20,
    display:"flex",alignItems:"center",gap:10,fontFamily:"'Space Grotesk',sans-serif"}}>{children}</h2>
);

const ScoreGauge = ({score}) => {
  const r=54,cx=64,cy=64,circ=2*Math.PI*r;
  const dash=(score/100)*circ;
  const color=score>75?C.green:score>25?C.amber:C.red;
  const label=score>75?"Optimale":score>25?"Normale":"Ecart";
  return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
      <svg width={128} height={128} viewBox="0 0 128 128">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.border} strokeWidth={10}/>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={10}
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round" transform="rotate(-90 64 64)"/>
        <text x={cx} y={cy-6} textAnchor="middle" fill={C.pri} fontSize={22}
          fontWeight={800} fontFamily="'Space Grotesk',sans-serif">{score.toFixed(0)}</text>
        <text x={cx} y={cy+14} textAnchor="middle" fill={C.sec} fontSize={11}>{label}</text>
      </svg>
    </div>
  );
};

// ── PAGE: Login ───────────────────────────────────────────────────────────────
const PageLogin = ({onLogin}) => {
  const [role,setRole]       = useState("praticien");
  const [user,setUser]       = useState("");
  const [pass,setPass]       = useState("");
  const [loading,setLoading] = useState(false);
  const [err,setErr]         = useState("");
  const [showPass,setShowPass] = useState(false);

  const CREDS = { praticien:{u:"praticien",p:"gait2024"}, chercheur:{u:"chercheur",p:"irl2024"} };

  const handleLogin = () => {
    setErr("");
    const ok = CREDS[role];
    if(user===ok.u && pass===ok.p){
      setLoading(true);
      setTimeout(()=>onLogin(role),1200);
    } else {
      setErr("Identifiants incorrects. Essayez : praticien / gait2024");
    }
  };

  // Animated background dots
  const dots = Array.from({length:30},(_,i)=>({
    cx:`${Math.random()*100}%`,
    cy:`${Math.random()*100}%`,
    r:Math.random()*2+1,
    opacity:Math.random()*.4+.1,
    animDelay:`${Math.random()*3}s`,
  }));

  return (
    <div style={{minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",
      position:"relative",overflow:"hidden",background:`radial-gradient(ellipse at 30% 50%, #0F1E3A 0%, ${C.bg} 60%)`}}>

      {/* Animated background */}
      <svg style={{position:"absolute",inset:0,width:"100%",height:"100%",pointerEvents:"none"}} xmlns="http://www.w3.org/2000/svg">
        {dots.map((d,i)=>(
          <circle key={i} cx={d.cx} cy={d.cy} r={d.r} fill="#3B82F6" opacity={d.opacity}
            style={{animation:`pulse 3s ${d.animDelay} ease-in-out infinite`}}/>
        ))}
        {/* Grid lines */}
        {Array.from({length:8},(_,i)=>(
          <line key={`h${i}`} x1="0" y1={`${i*14.3}%`} x2="100%" y2={`${i*14.3}%`}
            stroke="rgba(59,130,246,0.04)" strokeWidth="1"/>
        ))}
        {Array.from({length:12},(_,i)=>(
          <line key={`v${i}`} x1={`${i*9.1}%`} y1="0" x2={`${i*9.1}%`} y2="100%"
            stroke="rgba(59,130,246,0.04)" strokeWidth="1"/>
        ))}
        {/* Scanline */}
        <rect width="100%" height="3" fill="url(#scan)" style={{animation:"scanline 4s linear infinite"}}>
          <defs>
            <linearGradient id="scan" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="transparent"/>
              <stop offset="50%" stopColor="rgba(59,130,246,0.08)"/>
              <stop offset="100%" stopColor="transparent"/>
            </linearGradient>
          </defs>
        </rect>
      </svg>

      {/* Glow orbs */}
      <div style={{position:"absolute",top:"20%",left:"15%",width:300,height:300,
        background:"radial-gradient(circle,rgba(59,130,246,.12) 0%,transparent 70%)",
        borderRadius:"50%",filter:"blur(40px)",pointerEvents:"none"}}/>
      <div style={{position:"absolute",bottom:"20%",right:"15%",width:250,height:250,
        background:"radial-gradient(circle,rgba(139,92,246,.1) 0%,transparent 70%)",
        borderRadius:"50%",filter:"blur(40px)",pointerEvents:"none"}}/>

      {/* Left panel — branding */}
      <div className="fade-up" style={{position:"absolute",left:"8%",top:"50%",
        transform:"translateY(-50%)",maxWidth:380,display:"none"}} id="left-panel">
      </div>

      {/* Center card */}
      <div className="fade-up glass" style={{width:"100%",maxWidth:420,borderRadius:20,
        padding:"40px 36px",position:"relative",zIndex:10,
        boxShadow:"0 32px 64px rgba(0,0,0,.5), 0 0 0 1px rgba(255,255,255,.06)"}}>

        {/* Logo */}
        <div style={{textAlign:"center",marginBottom:32}}>
          <div className="float" style={{display:"inline-flex",alignItems:"center",
            justifyContent:"center",width:64,height:64,borderRadius:16,
            background:"linear-gradient(135deg,#1E3A5F,#1E1B4B)",
            border:"1px solid rgba(59,130,246,.3)",marginBottom:16,
            boxShadow:"0 0 32px rgba(59,130,246,.2)"}}>
            <svg width={32} height={32} viewBox="0 0 32 32" fill="none">
              <path d="M4 22 Q8 10 16 10 Q24 10 28 22" stroke="#3B82F6" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
              <circle cx="16" cy="10" r="3" fill="#3B82F6"/>
              <path d="M8 22 Q10 17 13 17 Q16 17 16 22" stroke="#14B8A6" strokeWidth="2" fill="none" strokeLinecap="round"/>
              <path d="M16 22 Q18 15 21 15 Q24 15 24 22" stroke="#8B5CF6" strokeWidth="2" fill="none" strokeLinecap="round"/>
            </svg>
          </div>
          <div style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:26,fontWeight:700,
            letterSpacing:-1,color:C.pri,lineHeight:1}}>
            Gait<span style={{color:C.accent}}>IRL</span>
          </div>
          <div style={{color:C.mut,fontSize:12,marginTop:6,letterSpacing:.5}}>
            Analyse biomecanique par renforcement inverse
          </div>
        </div>

        {/* Role selector */}
        <div style={{marginBottom:22}}>
          <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
            textTransform:"uppercase",marginBottom:10}}>Profil de connexion</div>
          <div style={{display:"flex",gap:8}}>
            {[
              {id:"praticien",label:"Praticien",icon:"⚕"},
              {id:"chercheur",label:"Chercheur",icon:"🔬"},
            ].map(r=>(
              <button key={r.id} className={`role-btn ${role===r.id?"active":""}`}
                onClick={()=>{setRole(r.id);setErr("");}}>
                {r.icon} {r.label}
              </button>
            ))}
          </div>
        </div>

        {/* Credentials hint */}
        <div style={{background:"rgba(59,130,246,.08)",border:"1px solid rgba(59,130,246,.15)",
          borderRadius:8,padding:"8px 12px",marginBottom:20,fontSize:12,color:C.sec}}>
          <strong style={{color:C.accent}}>Demo</strong> — {role==="praticien"?"praticien / gait2024":"chercheur / irl2024"}
        </div>

        {/* Fields */}
        <div style={{display:"flex",flexDirection:"column",gap:14,marginBottom:20}}>
          {/* Username */}
          <div style={{position:"relative"}}>
            <span style={{position:"absolute",left:14,top:"50%",transform:"translateY(-50%)",
              fontSize:16,pointerEvents:"none"}}>👤</span>
            <input className="input-field" placeholder="Identifiant" value={user}
              onChange={e=>{setUser(e.target.value);setErr("");}}
              onKeyDown={e=>e.key==="Enter"&&handleLogin()}/>
          </div>
          {/* Password */}
          <div style={{position:"relative"}}>
            <span style={{position:"absolute",left:14,top:"50%",transform:"translateY(-50%)",
              fontSize:16,pointerEvents:"none"}}>🔒</span>
            <input className="input-field" placeholder="Mot de passe"
              type={showPass?"text":"password"} value={pass}
              onChange={e=>{setPass(e.target.value);setErr("");}}
              onKeyDown={e=>e.key==="Enter"&&handleLogin()}
              style={{paddingRight:44}}/>
            <button onClick={()=>setShowPass(!showPass)}
              style={{position:"absolute",right:14,top:"50%",transform:"translateY(-50%)",
                background:"none",border:"none",cursor:"pointer",color:C.mut,fontSize:14}}>
              {showPass?"🙈":"👁"}
            </button>
          </div>
        </div>

        {/* Error */}
        {err && (
          <div style={{background:"rgba(239,68,68,.1)",border:"1px solid rgba(239,68,68,.3)",
            color:C.red,borderRadius:8,padding:"10px 14px",marginBottom:14,fontSize:13}}>
            {err}
          </div>
        )}

        {/* Submit */}
        <button className="login-btn glow-btn" onClick={handleLogin} disabled={loading}>
          {loading ? (
            <span style={{display:"flex",alignItems:"center",justifyContent:"center",gap:8}}>
              <span style={{width:16,height:16,border:"2px solid rgba(255,255,255,.3)",
                borderTopColor:"#fff",borderRadius:"50%",display:"inline-block",
                animation:"spin .8s linear infinite"}}/>
              Connexion en cours...
            </span>
          ) : "Acceder au tableau de bord"}
        </button>

        {/* Footer note */}
        <div style={{textAlign:"center",marginTop:20,color:C.mut,fontSize:11,lineHeight:1.6}}>
          Donnees medicales — acces confidentiel<br/>
          <span style={{color:C.accent,fontSize:10}}>Master 1 Machine Learning · M. Elfakir</span>
        </div>
      </div>
    </div>
  );
};

// ── PAGE: Accueil ─────────────────────────────────────────────────────────────
const PageAccueil = () => {
  const features = [
    {icon:"🦴",title:"Extraction Bio-mecanique",desc:"42 dimensions d'etat (angles, vitesses, GRF) + 18 dimensions d'action (couples articulaires) extraites depuis des capteurs MoCap a 100 Hz.",color:C.accent},
    {icon:"🧠",title:"Modelisation IRL (MaxEnt)",desc:"Reseau RewardNetwork 60→128→128→64→1 avec Dropout. Ratio R_expert/R_random = x3.07 apres 150 epochs de convergence.",color:C.teal},
    {icon:"📊",title:"Diagnostic Assiste par IA",desc:"Score percentile 0-100 compare a la distribution de reference. Pipeline video MediaPipe pour analyse en temps reel.",color:C.purple},
  ];

  return (
    <div>
      {/* Hero */}
      <div className="fade-up" style={{background:"linear-gradient(135deg,#0F1C3A 0%,#0B0F1A 60%)",
        border:`1px solid ${C.border}`,borderRadius:16,padding:"48px 40px",
        marginBottom:28,position:"relative",overflow:"hidden"}}>
        <div style={{position:"absolute",top:-80,right:-80,width:360,height:360,borderRadius:"50%",
          background:"radial-gradient(circle,rgba(59,130,246,.1) 0%,transparent 70%)",pointerEvents:"none"}}/>
        <div style={{position:"absolute",bottom:-40,left:-40,width:200,height:200,borderRadius:"50%",
          background:"radial-gradient(circle,rgba(139,92,246,.08) 0%,transparent 70%)",pointerEvents:"none"}}/>
        <div style={{color:C.accent,fontSize:11,fontWeight:700,letterSpacing:2,
          textTransform:"uppercase",marginBottom:14}}>GaitIRL — Biomecanique</div>
        <h1 style={{color:C.pri,fontSize:34,fontWeight:900,lineHeight:1.15,
          marginBottom:16,fontFamily:"'Space Grotesk',sans-serif",maxWidth:560}}>
          Apprentissage par<br/>
          <span style={{background:"linear-gradient(135deg,#3B82F6,#8B5CF6)",
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
            Renforcement Inverse
          </span><br/>
          de la Marche Humaine
        </h1>
        <p style={{color:C.sec,fontSize:14,maxWidth:500,lineHeight:1.75,marginBottom:28}}>
          Deduction de la fonction de recompense implicite optimisee par le corps humain,
          a partir de trajectoires biomecaniques observees sur 9 sujets sains.
        </p>
        <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
          {["MaxEnt IRL","190 618 frames","9 participants","251 trajectoires"].map(t=>(
            <span key={t} style={{background:"rgba(59,130,246,.1)",
              border:"1px solid rgba(59,130,246,.25)",color:C.accent,
              padding:"6px 14px",borderRadius:99,fontSize:12,fontWeight:600}}>{t}</span>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",
        gap:14,marginBottom:28}}>
        <StatCard label="R_expert / R_random" value="x3.07"  sub="Expert vs aleatoire" accent={C.accent}/>
        <StatCard label="Vitesse optimale"     value="1.2 m/s" sub="Decouverte spontanee" accent={C.teal}/>
        <StatCard label="Articulation dom."    value="Hanche" sub="32.5% - hierarchie confirmee" accent={C.green}/>
        <StatCard label="Loss finale"          value="-6.10"  sub="Convergence 150 epochs" accent={C.purple}/>
      </div>

      {/* Features */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(260px,1fr))",gap:16,marginBottom:28}}>
        {features.map((f,i)=>(
          <div key={i} className="hover-card" style={{background:C.card,
            border:`1px solid ${C.border}`,borderRadius:14,padding:24,
            animation:`fadeUp .6s ${i*.15}s cubic-bezier(.16,1,.3,1) both`}}>
            <div style={{width:44,height:44,borderRadius:12,
              background:`${f.color}18`,border:`1px solid ${f.color}33`,
              display:"flex",alignItems:"center",justifyContent:"center",
              fontSize:20,marginBottom:16}}>{f.icon}</div>
            <div style={{color:C.pri,fontWeight:700,fontSize:14,marginBottom:8,
              fontFamily:"'Space Grotesk',sans-serif"}}>{f.title}</div>
            <div style={{color:C.sec,fontSize:12,lineHeight:1.7}}>{f.desc}</div>
            <div style={{marginTop:14,height:2,borderRadius:1,background:`${f.color}40`}}/>
            <div style={{height:2,borderRadius:1,background:f.color,
              width:"60%",marginTop:-2,transition:"width .3s"}}/>
          </div>
        ))}
      </div>

      {/* Pipeline */}
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:26}}>
        <SectionTitle>Pipeline IRL — 6 phases</SectionTitle>
        <div style={{display:"flex",alignItems:"center",flexWrap:"wrap",gap:0}}>
          {[
            {n:"01",label:"Extraction",desc:"42 etats + 18 actions",color:C.accent},
            {n:"02",label:"Nettoyage",desc:"Butterworth 6 Hz",color:C.teal},
            {n:"03",label:"MaxEnt IRL",desc:"RewardNetwork",color:C.green},
            {n:"04",label:"Visualisation",desc:"7 figures poster",color:C.purple},
            {n:"05",label:"Vitesse",desc:"U inverse valide",color:C.amber},
            {n:"06",label:"Score",desc:"Percentile 0-100",color:C.red},
          ].map((s,i)=>(
            <div key={s.n} style={{display:"flex",alignItems:"center"}}>
              <div style={{textAlign:"center",padding:"0 6px"}}>
                <div style={{width:44,height:44,borderRadius:"50%",
                  background:`${s.color}18`,border:`2px solid ${s.color}`,
                  display:"flex",alignItems:"center",justifyContent:"center",
                  color:s.color,fontWeight:800,fontSize:12,margin:"0 auto 6px",
                  fontFamily:"'Space Grotesk',sans-serif"}}>{s.n}</div>
                <div style={{color:C.pri,fontWeight:700,fontSize:12}}>{s.label}</div>
                <div style={{color:C.mut,fontSize:10,marginTop:2}}>{s.desc}</div>
              </div>
              {i<5&&<div style={{width:20,height:1,background:C.border,flexShrink:0}}/>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ── PAGE: Analyse video ───────────────────────────────────────────────────────
const PageAnalyse = () => {
  const [dragging,setDragging] = useState(false);
  const [file,setFile]         = useState(null);
  const [loading,setLoading]   = useState(false);
  const [result,setResult]     = useState(null);
  const [err,setErr]           = useState(null);

  const handleFile = f => {
    if(!f) return;
    const ok = f.name.match(/\.(mp4|avi|mov)$/i);
    if(!ok){setErr("Format non supporte. Utilisez .mp4, .avi ou .mov");return;}
    setFile(f);setErr(null);setResult(null);
  };

  const analyze = async () => {
    if(!file) return;
    setLoading(true);setErr(null);
    const form = new FormData();
    form.append("video",file);
    try {
      const res = await fetch("http://localhost:5000/api/analyze-video",{method:"POST",body:form});
      if(!res.ok) throw new Error(`Erreur serveur ${res.status}`);
      setResult(await res.json());
    } catch(e) {
      setErr(e.message.includes("fetch")?"API Flask inaccessible — lancez python api.py":e.message);
    } finally { setLoading(false); }
  };

  const mockDemo = () => {
    setResult({
      score:42, niveau:"attention", r_moyen:30.52,
      rapport:"Legere asymetrie detectee a la hanche gauche. Vitesse angulaire cheville dans les normes. GRF symetrique.",
      angles_temporels:Array.from({length:40},(_,i)=>({
        t:i,
        hanche_g:15+20*Math.sin(i*.3)+Math.random()*3,
        genou_g:30+25*Math.sin(i*.3+1)+Math.random()*3,
        cheville_g:10+12*Math.sin(i*.3+2)+Math.random()*3,
      })),
      r_temporel:Array.from({length:40},(_,i)=>({t:i,r:30+1.5*Math.sin(i*.4)+Math.random()*.8})),
    });
  };

  return (
    <div>
      <SectionTitle>Analyse Video — Pipeline MediaPipe</SectionTitle>

      <div onDragOver={e=>{e.preventDefault();setDragging(true);}}
        onDragLeave={()=>setDragging(false)}
        onDrop={e=>{e.preventDefault();setDragging(false);handleFile(e.dataTransfer.files[0]);}}
        onClick={()=>document.getElementById("fi").click()}
        style={{border:`2px dashed ${dragging?C.accent:C.border}`,borderRadius:12,
          padding:"40px 24px",textAlign:"center",cursor:"pointer",
          background:dragging?"rgba(59,130,246,.05)":C.card,
          transition:"border-color .2s,background .2s",marginBottom:16}}>
        <input id="fi" type="file" accept=".mp4,.avi,.mov" hidden
          onChange={e=>handleFile(e.target.files[0])}/>
        <div style={{fontSize:36,marginBottom:10}}>📹</div>
        {file
          ? <div style={{color:C.pri,fontWeight:600}}>✅ {file.name}</div>
          : <div style={{color:C.sec}}>Glissez une video ici ou cliquez pour selectionner<br/>
              <span style={{fontSize:12,color:C.mut}}>Formats : .mp4 .avi .mov</span></div>
        }
      </div>

      {err&&<div style={{background:"rgba(239,68,68,.1)",border:"1px solid rgba(239,68,68,.3)",
        color:C.red,padding:"10px 14px",borderRadius:8,marginBottom:14,fontSize:13}}>⚠ {err}</div>}

      <div style={{display:"flex",gap:10,marginBottom:24}}>
        <button onClick={analyze} disabled={!file||loading} style={{
          background:C.accent,color:"#fff",border:"none",borderRadius:8,
          padding:"11px 22px",fontWeight:700,cursor:file&&!loading?"pointer":"not-allowed",
          opacity:file&&!loading?1:.5,fontSize:14,fontFamily:"'Inter',sans-serif"}}>
          {loading?"Analyse en cours...":"Analyser la video"}
        </button>
        <button onClick={mockDemo} style={{background:C.card,color:C.sec,
          border:`1px solid ${C.border}`,borderRadius:8,padding:"11px 22px",
          fontWeight:600,cursor:"pointer",fontSize:14,fontFamily:"'Inter',sans-serif"}}>
          Demo simulee
        </button>
      </div>

      {result&&(
        <div>
          <div style={{display:"grid",gridTemplateColumns:"auto 1fr",gap:16,marginBottom:16}}>
            <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,
              padding:20,textAlign:"center"}}>
              <ScoreGauge score={result.score}/>
              <div style={{marginTop:8}}><Badge niveau={result.niveau}/></div>
              <div style={{color:C.mut,fontSize:11,marginTop:6}}>R(s,a) = {result.r_moyen?.toFixed(2)}</div>
            </div>
            <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
              <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
                textTransform:"uppercase",marginBottom:10}}>Rapport automatique</div>
              <p style={{color:C.pri,lineHeight:1.7,fontSize:13}}>{result.rapport}</p>
              <div style={{marginTop:14,padding:"8px 12px",background:"rgba(59,130,246,.07)",
                borderRadius:8,fontSize:11,color:C.sec}}>
                ⚠ Score sur angles + vitesses uniquement (couples non disponibles depuis video). Resultat indicatif.
              </div>
            </div>
          </div>
          {result.angles_temporels&&(
            <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20,marginBottom:14}}>
              <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
                textTransform:"uppercase",marginBottom:14}}>Angles articulaires</div>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={result.angles_temporels}>
                  <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
                  <XAxis dataKey="t" stroke={C.mut} tick={{fontSize:10,fill:C.sec}}/>
                  <YAxis stroke={C.mut} tick={{fontSize:10,fill:C.sec}}/>
                  <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}/>
                  <Legend wrapperStyle={{fontSize:11}}/>
                  <Line type="monotone" dataKey="hanche_g" stroke={C.accent} dot={false} name="Hanche G" strokeWidth={2}/>
                  <Line type="monotone" dataKey="genou_g" stroke={C.teal} dot={false} name="Genou G" strokeWidth={2}/>
                  <Line type="monotone" dataKey="cheville_g" stroke={C.green} dot={false} name="Cheville G" strokeWidth={2}/>
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
          {result.r_temporel&&(
            <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
              <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
                textTransform:"uppercase",marginBottom:14}}>R(s,a) temporelle</div>
              <ResponsiveContainer width="100%" height={150}>
                <AreaChart data={result.r_temporel}>
                  <defs><linearGradient id="gR" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={C.accent} stopOpacity={.3}/>
                    <stop offset="95%" stopColor={C.accent} stopOpacity={0}/>
                  </linearGradient></defs>
                  <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
                  <XAxis dataKey="t" stroke={C.mut} tick={{fontSize:10,fill:C.sec}}/>
                  <YAxis stroke={C.mut} tick={{fontSize:10,fill:C.sec}} domain={["auto","auto"]}/>
                  <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}/>
                  <Area type="monotone" dataKey="r" stroke={C.accent} fill="url(#gR)" strokeWidth={2} dot={false} name="R(s,a)"/>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── PAGE: Participants ────────────────────────────────────────────────────────
const PageParticipants = () => {
  const [sel,setSel] = useState(null);
  const p = sel ? PARTICIPANTS.find(x=>x.id===sel) : null;
  return (
    <div>
      <SectionTitle>Participants — Leave-One-Out</SectionTitle>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(195px,1fr))",gap:12,marginBottom:24}}>
        {PARTICIPANTS.map(pt=>(
          <div key={pt.id} className="hover-card"
            onClick={()=>setSel(sel===pt.id?null:pt.id)}
            style={{background:sel===pt.id?`${C.accent}12`:C.card,
              border:`1px solid ${sel===pt.id?C.accent:C.border}`,
              borderRadius:12,padding:"16px 18px"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:10}}>
              <span style={{color:C.pri,fontWeight:800,fontSize:18,
                fontFamily:"'Space Grotesk',sans-serif"}}>{pt.id}</span>
              <Badge niveau={pt.niveau}/>
            </div>
            <div style={{display:"flex",justifyContent:"space-between"}}>
              <div><div style={{color:C.mut,fontSize:10,marginBottom:2}}>Score</div>
                <div style={{color:C.pri,fontWeight:700,fontSize:19,fontFamily:"'Space Grotesk',sans-serif"}}>{pt.score.toFixed(1)}</div></div>
              <div><div style={{color:C.mut,fontSize:10,marginBottom:2}}>R(s,a)</div>
                <div style={{color:C.accent,fontWeight:700,fontSize:14,fontFamily:"monospace"}}>{pt.r_moy.toFixed(2)}</div></div>
              <div><div style={{color:C.mut,fontSize:10,marginBottom:2}}>Cond.</div>
                <div style={{color:C.sec,fontWeight:700,fontSize:14}}>{pt.conditions}</div></div>
            </div>
            <div style={{marginTop:10,height:3,background:C.border,borderRadius:2,overflow:"hidden"}}>
              <div style={{height:"100%",borderRadius:2,
                background:pt.score>75?C.green:pt.score>25?C.amber:C.red,
                width:`${pt.score}%`,transition:"width .4s"}}/>
            </div>
          </div>
        ))}
      </div>

      {p&&(
        <div className="fade-in" style={{background:C.card,border:`1px solid ${C.border}`,
          borderRadius:12,padding:24,marginBottom:20}}>
          <div style={{display:"flex",alignItems:"center",gap:20,marginBottom:14}}>
            <ScoreGauge score={p.score}/>
            <div>
              <h3 style={{color:C.pri,fontWeight:800,fontSize:22,marginBottom:6,
                fontFamily:"'Space Grotesk',sans-serif"}}>{p.id}</h3>
              <Badge niveau={p.niveau}/>
              <div style={{marginTop:10,display:"grid",gridTemplateColumns:"1fr 1fr",gap:"6px 20px"}}>
                {[["R(s,a)",p.r_moy.toFixed(2)],["z-score",(p.z>=0?"+":"")+p.z.toFixed(3)],
                  ["Percentile",p.score.toFixed(1)],["Conditions",p.conditions]].map(([k,v])=>(
                  <div key={k}><span style={{color:C.mut,fontSize:11}}>{k} : </span>
                    <span style={{color:C.pri,fontWeight:700,fontFamily:"monospace",fontSize:12}}>{v}</span></div>
                ))}
              </div>
            </div>
          </div>
          <div style={{fontSize:12,color:C.sec,padding:"10px 14px",
            background:"rgba(59,130,246,.07)",borderRadius:8}}>
            z-score &lt; 0.25 — Participant homogene avec les autres sujets sains.
          </div>
        </div>
      )}

      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:22}}>
        <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
          textTransform:"uppercase",marginBottom:14}}>Comparaison R(s,a) par participant</div>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={PARTICIPANTS} barSize={28}>
            <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
            <XAxis dataKey="id" stroke={C.mut} tick={{fontSize:11,fill:C.sec}}/>
            <YAxis stroke={C.mut} tick={{fontSize:10,fill:C.sec}} domain={[30.2,31.0]}/>
            <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8,color:C.pri}}
              formatter={v=>[v.toFixed(3),"R(s,a)"]}/>
            <ReferenceLine y={30.58} stroke={C.mut} strokeDasharray="4 4"
              label={{value:"moyenne",fill:C.mut,fontSize:10}}/>
            <Bar dataKey="r_moy" fill={C.accent} radius={[4,4,0,0]}
              label={{position:"top",fill:C.mut,fontSize:9,formatter:v=>v.toFixed(2)}}/>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// ── PAGE: Reference ───────────────────────────────────────────────────────────
const PageReference = () => (
  <div>
    <SectionTitle>Modele de Reference — Distribution R(s,a)</SectionTitle>
    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:14,marginBottom:24}}>
      <StatCard label="Moyenne R(s,a)" value="30.58" sub="+-1.01 faible variance" accent={C.accent}/>
      <StatCard label="Vitesse optimale" value="1.2 m/s" sub="Polynome deg.2 -> 1.42 m/s" accent={C.teal}/>
      <StatCard label="R2 ajustement" value="0.479" sub="Courbe en U inverse" accent={C.green}/>
      <StatCard label="Trajectoires" value="251" sub="9 sujets x ~28 conditions" accent={C.purple}/>
    </div>

    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:22,marginBottom:16}}>
      <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
        textTransform:"uppercase",marginBottom:4}}>Distribution R(s,a) — Population de reference</div>
      <div style={{color:C.mut,fontSize:11,marginBottom:14}}>
        P5=29.27 · P25=30.38 · P50=30.75 · P75=31.02 · P95=31.50
      </div>
      <ResponsiveContainer width="100%" height={190}>
        <AreaChart data={DIST_REF}>
          <defs><linearGradient id="gD" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={C.accent} stopOpacity={.35}/>
            <stop offset="95%" stopColor={C.accent} stopOpacity={0}/>
          </linearGradient></defs>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
          <XAxis dataKey="x" stroke={C.mut} tick={{fontSize:10,fill:C.sec}}
            label={{value:"R(s,a)",position:"insideBottom",fill:C.mut,dy:6,fontSize:11}}/>
          <YAxis stroke={C.mut} tick={{fontSize:10,fill:C.sec}}/>
          <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}
            formatter={v=>[v.toFixed(4),"Densite"]}/>
          <ReferenceLine x="30.38" stroke={C.amber} strokeDasharray="3 3" label={{value:"P25",fill:C.amber,fontSize:10}}/>
          <ReferenceLine x="31.02" stroke={C.amber} strokeDasharray="3 3" label={{value:"P75",fill:C.amber,fontSize:10}}/>
          <Area type="monotone" dataKey="y" stroke={C.accent} fill="url(#gD)" strokeWidth={2} dot={false}/>
        </AreaChart>
      </ResponsiveContainer>
    </div>

    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:22,marginBottom:16}}>
      <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
        textTransform:"uppercase",marginBottom:4}}>R(s,a) vs Vitesse de marche — U inverse</div>
      <div style={{color:C.mut,fontSize:11,marginBottom:14}}>
        Vitesse optimale 1.2-1.4 m/s decouverte spontanement par le modele.
      </div>
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={VITESSE_DATA}>
          <CartesianGrid stroke={C.border} strokeDasharray="3 3"/>
          <XAxis dataKey="v" stroke={C.mut} tick={{fontSize:11,fill:C.sec}}
            label={{value:"Vitesse (m/s)",position:"insideBottom",fill:C.mut,dy:6,fontSize:11}}/>
          <YAxis stroke={C.mut} tick={{fontSize:11,fill:C.sec}} domain={[29.0,31.0]}/>
          <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}
            formatter={v=>[v.toFixed(2),"R(s,a)"]}/>
          <ReferenceLine x="1.2" stroke={C.green} strokeDasharray="4 4" label={{value:"Optimal",fill:C.green,fontSize:11}}/>
          <Line type="monotone" dataKey="r" stroke={C.teal} strokeWidth={2.5} dot={{fill:C.teal,r:4}} name="R(s,a)"/>
        </LineChart>
      </ResponsiveContainer>
    </div>

    <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
        <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
          textTransform:"uppercase",marginBottom:14}}>Importance par articulation</div>
        <ResponsiveContainer width="100%" height={190}>
          <PieChart>
            <Pie data={ARTICULAIRE_DATA} dataKey="pct" nameKey="name" cx="50%" cy="50%"
              outerRadius={75} label={({name,pct})=>`${name} ${pct}%`} labelLine={false}>
              {ARTICULAIRE_DATA.map((d,i)=><Cell key={i} fill={d.color}/>)}
            </Pie>
            <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}
              formatter={v=>[`${v}%`,"Contribution"]}/>
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
        <div style={{color:C.mut,fontSize:11,fontWeight:700,letterSpacing:1.2,
          textTransform:"uppercase",marginBottom:14}}>Importance par groupe de features</div>
        <ResponsiveContainer width="100%" height={190}>
          <BarChart data={IMPORTANCE_DATA} layout="vertical" barSize={11}>
            <CartesianGrid stroke={C.border} strokeDasharray="3 3" horizontal={false}/>
            <XAxis type="number" stroke={C.mut} tick={{fontSize:10,fill:C.sec}} domain={[0,20]}/>
            <YAxis type="category" dataKey="name" stroke={C.mut} tick={{fontSize:10,fill:C.sec}} width={100}/>
            <Tooltip contentStyle={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8}}
              formatter={v=>[`${v}%`,"Importance"]}/>
            <Bar dataKey="pct" radius={[0,4,4,0]}>
              {IMPORTANCE_DATA.map((d,i)=><Cell key={i} fill={d.color}/>)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div style={{marginTop:12,fontSize:11,color:C.mut,lineHeight:1.6}}>
          Methode : <span style={{color:C.sec,fontFamily:"monospace"}}>importance(i) = |dR/dxi| x |xi|</span>
          <br/>moyenne sur 18 000 frames (2 000 / participant)
        </div>
      </div>
    </div>
  </div>
);

// ── NAV icons ─────────────────────────────────────────────────────────────────
const NAV_ICONS = {
  accueil:      <svg width={15} height={15} viewBox="0 0 24 24" fill="currentColor"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>,
  analyse:      <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/><path d="M6 8l3 4 3-3 3 4"/></svg>,
  participants: <svg width={15} height={15} viewBox="0 0 24 24" fill="currentColor"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>,
  reference:    <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
};

const PAGES = [
  {id:"accueil",      label:"Accueil",       component:PageAccueil},
  {id:"analyse",      label:"Analyse video", component:PageAnalyse},
  {id:"participants", label:"Participants",  component:PageParticipants},
  {id:"reference",    label:"Reference",     component:PageReference},
];

// ── ROOT ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [role, setRole]         = useState(null);
  const [page, setPage]         = useState("accueil");
  const ActivePage = PAGES.find(p=>p.id===page)?.component || PageAccueil;

  // Inject global CSS once
  useEffect(()=>{
    const s = document.createElement("style");
    s.textContent = GLOBAL_CSS;
    document.head.appendChild(s);
    return ()=>document.head.removeChild(s);
  },[]);

  if(!loggedIn) return <PageLogin onLogin={r=>{setRole(r);setLoggedIn(true);}}/>;

  return (
    <div style={{minHeight:"100vh",background:C.bg,color:C.pri,
      fontFamily:"'Inter','Segoe UI',sans-serif"}}>

      {/* Nav */}
      <nav style={{background:`${C.surface}ee`,backdropFilter:"blur(12px)",
        WebkitBackdropFilter:"blur(12px)",borderBottom:`1px solid ${C.border}`,
        position:"sticky",top:0,zIndex:100}}>
        <div style={{maxWidth:1200,margin:"0 auto",padding:"0 24px",
          display:"flex",alignItems:"center",height:56,gap:6}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginRight:24}}>
            <div style={{width:28,height:28,borderRadius:8,
              background:"linear-gradient(135deg,#1E3A5F,#1E1B4B)",
              border:"1px solid rgba(59,130,246,.4)",display:"flex",
              alignItems:"center",justifyContent:"center"}}>
              <svg width={14} height={14} viewBox="0 0 32 32" fill="none">
                <path d="M4 22 Q8 10 16 10 Q24 10 28 22" stroke="#3B82F6" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
                <circle cx="16" cy="10" r="3" fill="#3B82F6"/>
              </svg>
            </div>
            <span style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:15,fontWeight:700,letterSpacing:-0.5}}>
              Gait<span style={{color:C.accent}}>IRL</span>
            </span>
          </div>

          {PAGES.map(p=>(
            <button key={p.id} className="nav-link" onClick={()=>setPage(p.id)} style={{
              background:page===p.id?`${C.accent}18`:"transparent",
              color:page===p.id?C.accent:C.sec,
              border:page===p.id?`1px solid ${C.accent}44`:"1px solid transparent",
              borderRadius:8,padding:"6px 12px",cursor:"pointer",
              fontWeight:page===p.id?700:500,fontSize:13,
              display:"flex",alignItems:"center",gap:6,
              fontFamily:"'Inter',sans-serif"}}>
              <span style={{color:page===p.id?C.accent:C.mut}}>{NAV_ICONS[p.id]}</span>
              {p.label}
            </button>
          ))}

          <div style={{marginLeft:"auto",display:"flex",alignItems:"center",gap:10}}>
            <div style={{padding:"4px 10px",background:`${C.accent}18`,
              border:`1px solid ${C.accent}33`,borderRadius:99,
              color:C.accent,fontSize:11,fontWeight:600,textTransform:"capitalize"}}>
              {role}
            </div>
            <button onClick={()=>setLoggedIn(false)} style={{background:"none",
              border:`1px solid ${C.border}`,borderRadius:8,padding:"4px 10px",
              color:C.mut,fontSize:12,cursor:"pointer",fontFamily:"'Inter',sans-serif"}}>
              Deconnexion
            </button>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main style={{maxWidth:1200,margin:"0 auto",padding:"28px 24px"}}>
        <ActivePage/>
      </main>

      {/* Footer */}
      <footer style={{borderTop:`1px solid ${C.border}`,padding:"16px 24px",
        textAlign:"center",color:C.mut,fontSize:11}}>
        GaitIRL — Biomecanique IRL · M. Elfakir ·{" "}
        <a href="https://figshare.com/articles/dataset/16530939" target="_blank" rel="noreferrer"
          style={{color:C.accent,textDecoration:"none"}}>Dataset Figshare</a>
      </footer>
    </div>
  );
}
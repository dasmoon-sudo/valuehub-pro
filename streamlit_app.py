import io, json, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objs as go
import streamlit as st, streamlit.components.v1 as components
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from math import isfinite

try:
    import yfinance as yf
except Exception:
    yf = None

# ---- TradingView ----
def tradingview_widget(tv_symbol="NASDAQ:NVDA", theme="light", height=520):
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_candle"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width":"100%","height":{height},"symbol":"{tv_symbol}","interval":"D",
        "timezone":"Etc/UTC","theme":"{theme}","style":"1","locale":"en",
        "withdateranges":true,"allow_symbol_change":true,"details":true,
        "container_id":"tv_candle"
      }});
      </script>
    </div>
    """
    components.html(html, height=height, scrolling=False)

def _krx_pad(code:str)->str:
    s=''.join(ch for ch in str(code) if ch.isdigit()); return s.zfill(6)

def make_tv_symbol(yf_ticker:str, info:Optional[pd.Series])->str:
    t = yf_ticker.upper()
    exch = None
    if isinstance(info, pd.Series):
        exch = (info.get("exchange") or info.get("market") or "").upper()
    if exch in ("NMS","NASDAQ"): return f"NASDAQ:{t}"
    if exch in ("NYQ","NYSE"):   return f"NYSE:{t}"
    if t.isdigit():              return f"KRX:{_krx_pad(t)}"
    return f"NASDAQ:{t}"

# ---- Utils ----
def _fmt_money(x: float, curr: str = "$") -> str:
    if pd.isna(x): return "-"
    try: return f"{curr}{x:,.0f}"
    except: return str(x)

# ---- Fetch ----
def fetch_yf(ticker: str) -> Dict[str, pd.DataFrame]:
    if yf is None: return {}
    t = yf.Ticker(ticker)
    try:
        price = t.history(period="1d")["Close"].iloc[-1]
    except Exception:
        price = np.nan
    info = pd.Series(getattr(t, "fast_info", {}))
    def _get(attr):
        df = getattr(t, attr, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty: return df
        try:
            if attr=="income_stmt": return t.get_income_stmt()
            if attr=="cashflow":    return t.get_cashflow()
            if attr=="balance_sheet": return t.get_balance_sheet()
        except Exception: pass
        return pd.DataFrame()
    return {"price": pd.Series([price], index=["last"]),
            "info": info,
            "income": _get("income_stmt"),
            "cash": _get("cashflow"),
            "balance": _get("balance_sheet")}

def fetch_price_history(ticker: str, period="5y", interval="1wk")->Optional[pd.Series]:
    if yf is None: return None
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist is not None and not hist.empty and "Close" in hist:
            return hist["Close"].dropna()
    except Exception: pass
    return None

def calc_cagr(series: pd.Series) -> float:
    if series is None or series.empty or len(series)<2: return float("nan")
    a, b = series.iloc[0], series.iloc[-1]
    if not (isfinite(a) and isfinite(b)) or a<=0: return float("nan")
    years = max((series.index[-1]-series.index[0]).days/365.25, 1e-9)
    return (b/a)**(1/years) - 1

def returns_stats(series: pd.Series)->Dict[str,float]:
    if series is None or series.empty: return {"mean_daily":np.nan,"ann_vol":np.nan}
    r = np.log(series).diff().dropna()
    return {"mean_daily": float(r.mean()), "ann_vol": float(r.std()*np.sqrt(252))}

# ---- Models ----
@dataclass
class CapmInputs:
    rf: float=0.04; beta: float=1.2; erp: float=0.05; cs: float=0.0
    @property
    def cost_of_equity(self)->float: return self.rf + self.beta*(self.erp+self.cs)

@dataclass
class WaccInputs:
    cost_of_equity: float; cost_of_debt_pre_tax: float; tax_rate: float
    equity_weight: float; debt_weight: float
    @property
    def wacc(self)->float:
        ce=self.cost_of_equity; cd=self.cost_of_debt_pre_tax*(1-self.tax_rate)
        ew, dw = self.equity_weight, self.debt_weight
        if ew+dw==0: return ce
        ew, dw = ew/(ew+dw), dw/(ew+dw)
        return ew*ce + dw*cd

@dataclass
class ForecastDrivers:
    horizon_years:int=10
    revenue_start:float=1e9
    revenue_cagr:float=0.10
    op_margin_start:float=0.28
    op_margin_target:float=0.32
    margin_years_to_target:int=5
    tax_rate:float=0.18
    dep_as_pct_rev:float=0.03
    capex_as_pct_rev:float=0.04
    wc_as_pct_rev:float=0.03

@dataclass
class TerminalValue:
    method:str="gordon"            # "gordon" | "exit_multiple"
    perpetuity_growth:float=0.025
    exit_multiple_ebit:float=20.0

@dataclass
class ShareInfo:
    net_debt:float=0.0
    shares_out:float=2_500_000_000
    sbc_as_pct_rev:float=0.0

@dataclass
class DcfResult:
    implied_ev:float; implied_equity:float; implied_price:float
    pv_fcf:float; pv_tv:float; wacc:float

# ---- Forecast engine ----
def linear_to_target(start:float, target:float, years:int, horizon:int)->List[float]:
    arr=[]
    for y in range(1, horizon+1):
        if years<=1: val=target
        else:
            w=min(y/years,1.0); val=start*(1-w)+target*w
        arr.append(val)
    return arr

def build_forecast(dr: ForecastDrivers)->pd.DataFrame:
    yrs=list(range(1, dr.horizon_years+1))
    rev=[dr.revenue_start*((1+dr.revenue_cagr)**y) for y in yrs]
    opm=linear_to_target(dr.op_margin_start, dr.op_margin_target, dr.margin_years_to_target, dr.horizon_years)
    ebit=[r*m for r,m in zip(rev, opm)]
    taxes=[max(e,0)*dr.tax_rate for e in ebit]
    nopat=[e-t for e,t in zip(ebit, taxes)]
    dep=[r*dr.dep_as_pct_rev for r in rev]
    capex=[r*dr.capex_as_pct_rev for r in rev]
    wc=[r*dr.wc_as_pct_rev for r in rev]
    dwc=[wc[0]]+[max(wc[i]-wc[i-1],0.0) for i in range(1,len(wc))]
    fcf=[n+d-c-w for n,d,c,w in zip(nopat,dep,capex,dwc)]
    return pd.DataFrame({"Year":yrs,"Revenue":rev,"OpMargin":opm,"EBIT":ebit,"NOPAT":nopat,"D&A":dep,"Capex":capex,"WC":wc,"ΔWC":dwc,"FCF":fcf})

# ---- DCF ----
def present_value(cashflows: List[float], rate: float)->float:
    return sum(cf/((1+rate)**(i+1)) for i,cf in enumerate(cashflows))

def terminal_value(df: pd.DataFrame, tv: TerminalValue, wacc: float)->Tuple[float,float]:
    last_fcf=df["FCF"].iloc[-1]
    if tv.method=="gordon":
        g=tv.perpetuity_growth
        tv_cash= last_fcf*(1+g)/(wacc-g)
    else:
        tv_cash= df["EBIT"].iloc[-1]*tv.exit_multiple_ebit
    pv = tv_cash/((1+wacc)**len(df))
    return tv_cash, pv

def run_dcf(df: pd.DataFrame, wacc: float, tv: TerminalValue, shares: ShareInfo)->DcfResult:
    pv_fcf = present_value(df["FCF"].tolist(), wacc)
    tv_cash, pv_tv = terminal_value(df, tv, wacc)
    implied_ev = pv_fcf + pv_tv
    implied_equity = implied_ev - shares.net_debt
    price = implied_equity/max(shares.shares_out,1e-9)
    return DcfResult(implied_ev, implied_equity, price, pv_fcf, pv_tv, wacc)

# ---- App ----
st.set_page_config(page_title="ValueHub Pro", layout="wide")
st.sidebar.title("ValueHub Pro")

# Sidebar
ticker = st.sidebar.text_input("Ticker (NVDA / 005930 등)", value="NVDA")
ph_period = st.sidebar.selectbox("Price period", ["1y","3y","5y","10y","max"], index=2)
ph_interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=1)
apply_from_price = st.sidebar.checkbox("가격 CAGR → 매출 CAGR 반영", value=False)
map_mult = st.sidebar.slider("매핑 배수 (Price→Revenue)", 0.2, 1.5, 1.0, 0.05)

currency = st.sidebar.selectbox("통화", ["$","₩","€"], index=0)
shares_out = st.sidebar.number_input("발행주식수", value=2_500_000_000, step=1_000_000)
net_debt = st.sidebar.number_input("순부채", value=0.0, step=1_000_000.0)

rf = st.sidebar.number_input("rf", value=0.04, step=0.005, format="%f")
beta = st.sidebar.number_input("beta", value=1.20, step=0.05, format="%f")
erp = st.sidebar.number_input("ERP", value=0.05, step=0.005, format="%f")
cs = st.sidebar.number_input("Country/Other Spread", value=0.00, step=0.005, format="%f")
coed = CapmInputs(rf,beta,erp,cs).cost_of_equity
cost_of_debt = st.sidebar.number_input("세전 부채비용", value=0.05, step=0.005, format="%f")
tax_rate = st.sidebar.number_input("법인세율", value=0.18, step=0.01, format="%f")
e_weight = st.sidebar.slider("Equity Weight", 0.0, 1.0, 0.95, 0.01)
wacc_val = WaccInputs(coed, cost_of_debt, tax_rate, e_weight, 1.0-e_weight).wacc
st.sidebar.write(f"**WACC ≈ {wacc_val*100:.2f}%**")

horizon = st.sidebar.slider("Forecast Horizon (Years)", 5, 20, 10)
terminal_method = st.sidebar.selectbox("Terminal Method", ["gordon","exit_multiple"], index=0)
terminal_g = st.sidebar.number_input("Perpetuity g", value=0.025, step=0.002, format="%f")
exit_mult = st.sidebar.number_input("Exit Multiple (EBIT)", value=20.0, step=1.0)
sbc_pct = st.sidebar.slider("SBC as % of Revenue (cash 처럼 차감)", 0.00, 0.20, 0.00, 0.005)

tv_theme = st.sidebar.selectbox("TradingView Theme", ["light","dark"], index=0)
tv_height = st.sidebar.slider("TV Height", 360, 900, 520, 10)

fetch_btn = st.sidebar.button("Fetch / 불러오기")

# Header
c1,c2 = st.columns([0.7,0.3])
with c1:
    st.title("📈 ValueHub Pro — DCF & Forecasting")
    st.caption("데이터 기반 예측 · 컨센서스 오버레이 · Reverse DCF · 민감도 · Monte Carlo")
with c2:
    st.metric("WACC", f"{wacc_val*100:.2f}%")
    st.metric("CoE", f"{coed*100:.2f}%")

# Fetch data
market_data = {}
price_hist = None
if fetch_btn and ticker:
    with st.spinner("Fetching market & financials from yfinance…"):
        try:
            market_data = fetch_yf(ticker)
            price_hist = fetch_price_history(ticker, ph_period, ph_interval)
        except Exception as e:
            st.warning(f"Market fetch failed: {e}")

if market_data:
    try:
        last_px = market_data.get("price", pd.Series([np.nan]))[0]
    except Exception:
        last_px = np.nan
    st.success(f"Fetched data for **{ticker}**")
    st.metric("Last Price", f"{currency}{last_px:,.2f}" if not pd.isna(last_px) else "-")

# Prepare initial forecast (so other tabs can use it)
rev_start_default = 100_000_000_000.0
rev_cagr_default = 0.10
drivers = ForecastDrivers(
    horizon_years=horizon, revenue_start=rev_start_default, revenue_cagr=rev_cagr_default,
    op_margin_start=0.28, op_margin_target=0.32, margin_years_to_target=5,
    tax_rate=tax_rate, dep_as_pct_rev=0.03, capex_as_pct_rev=0.04, wc_as_pct_rev=0.03
)
df_forecast = build_forecast(drivers)

# Tabs
tabs = st.tabs(["Price", "Live", "Forecast", "Consensus", "DCF", "Sensitivity", "Monte Carlo", "Scenarios"])

# Price tab
with tabs[0]:
    if price_hist is not None and not price_hist.empty:
        st.line_chart(price_hist)
        cagr = calc_cagr(price_hist); stats = returns_stats(price_hist)
        cA,cB,cC = st.columns(3)
        cA.metric("Price CAGR (period)", f"{cagr*100:.2f}%")
        cB.metric("Ann. Vol (est)", f"{stats['ann_vol']*100:.1f}%")
        cC.metric("Mean daily log-ret", f"{stats['mean_daily']*100:.2f}%")
        if apply_from_price and isfinite(cagr):
            st.session_state['rev_cagr'] = float(np.clip(cagr*map_mult, 0.0, 0.60))
            st.success(f"Revenue CAGR 슬라이더에 {st.session_state['rev_cagr']*100:.2f}% 적용됨")
    else:
        st.info("사이드바에서 티커·기간·빈도를 선택하고 Fetch를 눌러주세요.")

# Live tab
with tabs[1]:
    tv_symbol = make_tv_symbol(ticker, market_data.get("info", None) if market_data else None)
    st.caption(f"TradingView symbol: **{tv_symbol}**")
    tradingview_widget(tv_symbol, theme=tv_theme, height=tv_height)

# Forecast tab
with tabs[2]:
    rev_start = st.number_input("Revenue Start (다음 해)", value=rev_start_default, step=10_000_000.0)
    rev_cagr = st.slider("Revenue CAGR", 0.00, 0.60, st.session_state.get("rev_cagr", rev_cagr_default), 0.01)
    opm_start = st.slider("Op Margin (Start)", 0.00, 0.60, 0.28, 0.01)
    opm_target = st.slider("Op Margin (Target)", 0.00, 0.60, 0.32, 0.01)
    years_to_target = st.slider("Years to Target", 1, 15, 5)
    dep_pct = st.slider("D&A as % of Rev", 0.00, 0.20, 0.03, 0.005)
    capex_pct = st.slider("Capex as % of Rev", 0.00, 0.30, 0.04, 0.005)
    wc_pct = st.slider("WC as % of Rev", 0.00, 0.30, 0.03, 0.005)

    drivers = ForecastDrivers(
        horizon_years=horizon, revenue_start=rev_start, revenue_cagr=rev_cagr,
        op_margin_start=opm_start, op_margin_target=opm_target, margin_years_to_target=years_to_target,
        tax_rate=tax_rate, dep_as_pct_rev=dep_pct, capex_as_pct_rev=capex_pct, wc_as_pct_rev=wc_pct
    )
    df_forecast = build_forecast(drivers)
    if sbc_pct>0:
        df_forecast["SBC"] = df_forecast["Revenue"]*sbc_pct
        df_forecast["FCF"] = df_forecast["FCF"] - df_forecast["SBC"]

    st.dataframe(df_forecast.style.format({
        "Revenue": lambda x:_fmt_money(x,currency),
        "EBIT": lambda x:_fmt_money(x,currency),
        "NOPAT": lambda x:_fmt_money(x,currency),
        "D&A": lambda x:_fmt_money(x,currency),
        "Capex": lambda x:_fmt_money(x,currency),
        "WC": lambda x:_fmt_money(x,currency),
        "ΔWC": lambda x:_fmt_money(x,currency),
        "FCF": lambda x:_fmt_money(x,currency),
    }))
    st.plotly_chart(px.line(df_forecast, x="Year", y=["Revenue","EBIT","FCF"], title="Revenue / EBIT / FCF"), use_container_width=True)

# Consensus tab
with tabs[3]:
    st.caption("연간/분기 컨센서스 또는 가이던스를 CSV로 업로드하세요.")
    cons_annual = st.file_uploader("Annual Consensus CSV (Year, Revenue, EBIT, EPS, FCF)", type=["csv"], key="cons_a")
    cons_quarter = st.file_uploader("Quarterly Consensus CSV (Period, Revenue, EBIT, EPS, FCF)", type=["csv"], key="cons_q")

    annual_df = None; quarter_df=None
    if cons_annual is not None:
        try: annual_df = pd.read_csv(cons_annual); st.success("Annual consensus loaded")
        except Exception as e: st.error(f"Annual CSV parse failed: {e}")

    if cons_quarter is not None:
        try:
            quarter_df = pd.read_csv(cons_quarter)
            if 'Period' in quarter_df.columns:
                def _parse_period(p):
                    p=str(p)
                    if 'Q' in p:
                        y,q=p.split('Q'); q=int(q)
                        month={1:3,2:6,3:9,4:12}.get(q,3)
                        return pd.Timestamp(int(y), month, 1)
                    try: return pd.to_datetime(p)
                    except: return pd.NaT
                quarter_df['Date']=quarter_df['Period'].apply(_parse_period)
            st.success("Quarterly consensus loaded")
        except Exception as e: st.error(f"Quarterly CSV parse failed: {e}")

    # Model vs Consensus Overlay (NAN-safe)
    overlay_tabs = st.tabs(["Revenue", "EBIT", "EPS", "FCF"])

    def _safe_series(df, col):
        try:
            if df is None or col not in df.columns:
                return None
            s = pd.to_numeric(df[col], errors="coerce")
            return s if s.notna().any() else None
        except Exception:
            return None

    for i, metric in enumerate(["Revenue", "EBIT", "EPS", "FCF"]):
        with overlay_tabs[i]:
            fig = go.Figure()

            # Model line (존재할 때만)
            y_model = _safe_series(df_forecast, metric)
            if y_model is not None:
                fig.add_trace(go.Scatter(
                    x=df_forecast["Year"], y=y_model, mode="lines+markers", name="Model"
                ))

            # Annual consensus
            if annual_df is not None and {"Year", metric}.issubset(set(annual_df.columns)):
                ya = _safe_series(annual_df, metric)
                if ya is not None:
                    xa = annual_df["Year"]
                    if len(ya) == len(xa):
                        fig.add_trace(go.Scatter(
                            x=xa, y=ya, mode="lines+markers", name="Consensus (Annual)"
                        ))

            # Quarterly consensus (연도 합산)
            if quarter_df is not None and metric in (quarter_df.columns if quarter_df is not None else []):
                qtmp = quarter_df.copy()
                if "Date" in qtmp.columns and qtmp["Date"].notna().any():
                    qtmp["Year"] = qtmp["Date"].dt.year
                elif "Period" in qtmp.columns:
                    qtmp["Year"] = qtmp["Period"].astype(str).str.slice(0,4).astype(int)
                if "Year" in qtmp.columns:
                    agg = qtmp.groupby("Year")[metric].sum(numeric_only=True).reset_index()
                    yaq = pd.to_numeric(agg[metric], errors="coerce")
                    if yaq.notna().any():
                        fig.add_trace(go.Bar(
                            x=agg["Year"], y=yaq, name="Consensus (Quarter agg)", opacity=0.5
                        ))

            if not fig.data:
                st.info(f"**{metric}** 데이터가 없어 그래프를 표시할 수 없어요.")
            else:
                fig.update_layout(title=f"{metric} — Model vs Consensus",
                                  xaxis_title="Year", yaxis_title=metric)
                st.plotly_chart(fig, use_container_width=True)

# DCF tab (+ Reverse DCF)
with tabs[4]:
    st.subheader("DCF & Reverse DCF")
    shares = ShareInfo(net_debt=net_debt, shares_out=shares_out, sbc_as_pct_rev=sbc_pct)
    tv_inputs = TerminalValue(method=terminal_method, perpetuity_growth=terminal_g, exit_multiple_ebit=exit_mult)
    res = run_dcf(df_forecast, wacc_val, tv_inputs, shares)

    # Reverse DCF
    last_px = market_data.get("price", pd.Series([np.nan]))[0] if market_data else np.nan
    default_tp = float(last_px) if isfinite(last_px) else 100.0
    target_price = st.number_input("목표 주가", value=default_tp)

    def solve_rev_cagr_from_price(target_px: float, low=0.0, high=0.60, tol=1e-4, maxit=60)->float:
        lo,hi=low,high
        for _ in range(maxit):
            mid=0.5*(lo+hi)
            d=ForecastDrivers(
                horizon_years=horizon, revenue_start=df_forecast["Revenue"].iloc[0]/(1+drivers.revenue_cagr),
                revenue_cagr=mid, op_margin_start=drivers.op_margin_start,
                op_margin_target=drivers.op_margin_target, margin_years_to_target=drivers.margin_years_to_target,
                tax_rate=tax_rate, dep_as_pct_rev=drivers.dep_as_pct_rev, capex_as_pct_rev=drivers.capex_as_pct_rev,
                wc_as_pct_rev=drivers.wc_as_pct_rev
            )
            df=build_forecast(d)
            if sbc_pct>0: df["FCF"]=df["FCF"]-(df["Revenue"]*sbc_pct)
            r=run_dcf(df, wacc_val, tv_inputs, shares)
            if r.implied_price>target_px: hi=mid
            else: lo=mid
            if abs(r.implied_price-target_px)/max(target_px,1e-6)<tol: return mid
        return 0.5*(lo+hi)

    if st.button("Solve Revenue CAGR from Price"):
        solved=solve_rev_cagr_from_price(float(target_price))
        st.session_state['rev_cagr']=float(np.clip(solved,0.0,0.60))
        st.success(f"Solved Revenue CAGR ≈ {st.session_state['rev_cagr']*100:.2f}% (슬라이더에 적용됨)")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("PV(FCF)", _fmt_money(res.pv_fcf, currency))
    c2.metric("PV(Terminal)", _fmt_money(res.pv_tv, currency))
    c3.metric("Equity Value", _fmt_money(res.implied_equity, currency))
    c4.metric("Implied Price", f"{currency}{res.implied_price:,.2f}")

    mos = st.slider("Margin of Safety %", 0, 80, 25)
    st.info(f"**MOS Price:** {currency}{res.implied_price*(1-mos/100):,.2f}")

# Sensitivity
with tabs[5]:
    st.subheader("Sensitivity — WACC vs g / Exit Multiple")
    if terminal_method=="gordon":
        wmin,wmax = st.slider("WACC Range %", 4.0, 18.0, (max(1.0,wacc_val*100-2), min(18.0,wacc_val*100+2)))
        gmin,gmax = st.slider("g Range %", 0.0, 6.0, (max(0.0, terminal_g*100-1), min(6.0, terminal_g*100+1)))
        wacc_list = np.round(np.linspace(wmin/100, wmax/100, 7), 4).tolist()
        g_list = np.round(np.linspace(gmin/100, gmax/100, 7), 4).tolist()
        data=[]
        for w in wacc_list:
            row=[]
            for g in g_list:
                tv=TerminalValue(method="gordon", perpetuity_growth=g)
                r=run_dcf(df_forecast, w, tv, ShareInfo(net_debt=net_debt, shares_out=shares_out, sbc_as_pct_rev=sbc_pct))
                row.append(r.implied_price)
            data.append(row)
        sens=pd.DataFrame(data, index=[f"{w*100:.1f}%" for w in wacc_list], columns=[f"{g*100:.1f}%" for g in g_list])
        sens.index.name="WACC"; sens.columns.name="Terminal g"
    else:
        wmin,wmax = st.slider("WACC Range %", 4.0, 18.0, (max(1.0,wacc_val*100-2), min(18.0,wacc_val*100+2)))
        mmin,mmax = st.slider("Exit Multiple Range", 8.0, 40.0, (12.0, 30.0))
        wacc_list=np.round(np.linspace(wmin/100, wmax/100, 7),4).tolist()
        mult_list=np.round(np.linspace(mmin, mmax, 7),2).tolist()
        data=[]
        for w in wacc_list:
            row=[]
            for m in mult_list:
                tv=TerminalValue(method="exit_multiple", exit_multiple_ebit=m)
                r=run_dcf(df_forecast, w, tv, ShareInfo(net_debt=net_debt, shares_out=shares_out, sbc_as_pct_rev=sbc_pct))
                row.append(r.implied_price)
            data.append(row)
        sens=pd.DataFrame(data, index=[f"{w*100:.1f}%" for w in wacc_list], columns=[f"x{m:.1f}" for m in mult_list])
        sens.index.name="WACC"; sens.columns.name="Exit Mult (EBIT)"
    st.dataframe(sens.style.format(lambda x: f"{currency}{x:,.2f}"))
    st.plotly_chart(px.imshow(sens.values, x=list(sens.columns), y=list(sens.index), aspect="auto",
                              title="Sensitivity Heatmap (Price)"), use_container_width=True)

# Monte Carlo
with tabs[6]:
    st.subheader("Monte Carlo (1000 sims)")
    mc_on = st.checkbox("Run", value=False)
    if mc_on:
        w_mu = st.number_input("WACC mean", value=float(wacc_val))
        w_sd = st.number_input("WACC stdev", value=0.01, step=0.005, format="%f")
        g_mu = st.number_input("g mean", value=float(terminal_g))
        g_sd = st.number_input("g stdev", value=0.005, step=0.002, format="%f")
        prices=[]
        rng=np.random.default_rng(42)
        for _ in range(1000):
            w=max(0.0, rng.normal(w_mu, w_sd))
            g=rng.normal(g_mu, g_sd); g=min(g, w-0.002) if w>0.002 else g
            r=run_dcf(df_forecast, w, TerminalValue(method="gordon", perpetuity_growth=g),
                      ShareInfo(net_debt=net_debt, shares_out=shares_out, sbc_as_pct_rev=sbc_pct))
            prices.append(r.implied_price)
        sims=pd.Series(prices)
        st.write(sims.describe())
        st.plotly_chart(px.histogram(sims, nbins=50, title="Implied Price Distribution"), use_container_width=True)

# Scenarios
with tabs[7]:
    st.subheader("Scenarios — 저장/불러오기")
    if "scenarios" not in st.session_state: st.session_state.scenarios={}
    sc_name = st.text_input("Scenario name", value="Base")
    if st.button("Save Scenario"):
        pkg = {
            "drivers": asdict(ForecastDrivers(
                horizon_years=horizon, revenue_start=rev_start if 'rev_start' in locals() else 100e9,
                revenue_cagr=st.session_state.get('rev_cagr', 0.10),
                op_margin_start=opm_start if 'opm_start' in locals() else 0.28,
                op_margin_target=opm_target if 'opm_target' in locals() else 0.32,
                margin_years_to_target=years_to_target if 'years_to_target' in locals() else 5,
                tax_rate=tax_rate, dep_as_pct_rev=dep_pct if 'dep_pct' in locals() else 0.03,
                capex_as_pct_rev=capex_pct if 'capex_pct' in locals() else 0.04,
                wc_as_pct_rev=wc_pct if 'wc_pct' in locals() else 0.03
            )),
            "wacc": float(wacc_val),
            "capm": {"rf": rf, "beta": beta, "erp": erp, "cs": cs},
            "terminal": {"method": terminal_method, "g": terminal_g, "exit_mult": exit_mult},
            "shares": {"net_debt": net_debt, "shares_out": shares_out, "sbc_pct": sbc_pct},
        }
        st.session_state.scenarios[sc_name]=pkg
        st.success(f"Saved scenario '{sc_name}'.")
    if st.session_state.scenarios:
        sc_pick = st.selectbox("Load Scenario", list(st.session_state.scenarios.keys()))
        if st.button("Download JSON"):
            buf=io.StringIO(); json.dump(st.session_state.scenarios[sc_pick], buf, indent=2)
            st.download_button("Download", data=buf.getvalue(), file_name=f"{sc_pick}.json", mime="application/json")
        st.json(st.session_state.scenarios)

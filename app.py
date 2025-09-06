import io, math
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_EVEN
from dateutil.relativedelta import relativedelta
import calendar
import pandas as pd
import streamlit as st

# ---------------- Utils ----------------
def is_leap(y:int)->bool:
    return calendar.isleap(y)

def _q2(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)

def days_in_year(d:date)->int:
    return 366 if is_leap(d.year) else 365

def month_start(d:date)->date:
    return d.replace(day=1)

def month_end(d:date)->date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])

def overlap_days(a0:date,a1:date,b0:date,b1:date)->int:
    s=max(a0,b0); e=min(a1,b1)
    return 0 if s>e else (e-s).days+1

def build_months(a:date,b:date):
    out=[]; cur=month_start(a); end=month_end(b)
    while cur<=end:
        out.append((month_start(cur), month_end(cur)))
        cur += relativedelta(months=1)
    return out

# ---------------- Engine ----------------
def generate_schedule(
    cost: float,
    salvage: float,
    purchase_dt: date,
    method: str,          # 'SLM' or 'WDV'
    rate_pct: float,      # annual rate percent (required)
    convention: str       # 'full-month' or 'exact-days'
) -> pd.DataFrame:

    if cost <= 0:
        raise ValueError("Cost must be > 0")
    if salvage < 0:
        raise ValueError("Salvage cannot be negative")
    if rate_pct <= 0:
        raise ValueError("Rate % must be > 0")
    if method == 'SLM' and salvage >= cost:
        raise ValueError("For SLM, salvage must be less than cost.")

    # End date: for SLM run full life, for WDV cap at 30y
    if method == 'SLM':
        months_life = max(1, math.ceil(12.0 / (rate_pct / 100.0)))
        pro_end = purchase_dt + relativedelta(months=months_life) - (
            timedelta(days=1) if convention == 'exact-days' else timedelta(0)
        )
    else:
        pro_end = purchase_dt + relativedelta(years=30)

    months = build_months(purchase_dt, pro_end)

    rows = []
    opening = float(cost)
    dep_base_slm_annual = (cost - salvage) * (rate_pct / 100.0)

    for (m0, m1) in months:
        # Convention proration
        if convention == 'full-month':
            eligible = m1 >= purchase_dt
            pror = (1.0 / 12.0) if eligible else 0.0
        else:
            od = overlap_days(purchase_dt, pro_end, m0, m1)
            eligible = od > 0
            pror = (od / days_in_year(m0)) if eligible else 0.0

        if not eligible:
            continue

        # Annual basis
        if method == 'SLM':
            annual = dep_base_slm_annual
        else:  # WDV
            annual = opening * (rate_pct / 100.0)

        computed_charge = annual * pror

        # Salvage cap
        headroom = max(0.0, opening - salvage)
        dep = min(headroom, computed_charge)

        opening_q = float(_q2(opening))
        dep_q     = float(_q2(dep))
        closing   = opening_q - dep_q
        closing_q = float(_q2(closing))

        rows.append({
            "Period": m0.strftime("%Y-%m"),
            "Period Start": m0,
            "Period End": m1,
            "Opening NBV": opening_q,
            "Depreciation": dep_q,
            "Closing NBV": closing_q,
        })

        opening = closing_q
        if opening <= salvage + 1e-6:
            break

    # Balancing
    df = pd.DataFrame(rows)
    if not df.empty:
        total_dep = sum(_q2(x) for x in df["Depreciation"])
        target    = _q2(cost - salvage)
        diff      = total_dep - target
        if diff != Decimal("0.00"):
            for i in range(len(df)-1, -1, -1):
                cur_dep = _q2(df.at[i, "Depreciation"])
                if cur_dep > Decimal("0.00"):
                    desired = cur_dep - diff
                    if desired < Decimal("0.00"):
                        desired = Decimal("0.00")
                    delta = desired - cur_dep
                    df.at[i, "Depreciation"] = float(_q2(desired))
                    df.at[i, "Closing NBV"]  = float(_q2(Decimal(str(df.at[i, "Closing NBV"])) - delta))
                    break

    _validate_schedule(df, cost, salvage)
    return df

# ---------------- Validation ----------------
def _validate_schedule(df: pd.DataFrame, cost: float, salvage: float) -> None:
    if df.empty:
        return
    closing_nbv = _q2(df.iloc[-1]["Closing NBV"])
    total_dep   = sum(_q2(v) for v in df["Depreciation"])
    max_dep     = _q2(cost - salvage)
    assert closing_nbv >= _q2(salvage) - Decimal("0.01"), (
        f"Final NBV {closing_nbv} fell below salvage {salvage}"
    )
    assert total_dep <= max_dep + Decimal("0.01"), (
        f"Total depreciation {total_dep} exceeds depreciable base {max_dep}"
    )

def df_to_pdf_bytes(header, df):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=24, rightMargin=24, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph("<b>Depreciation Schedule</b>", styles["Title"]), Spacer(1,6)]
    meta = [f"<b>{k}:</b> {v}" for k,v in header.items()]
    story += [Paragraph("<br/>".join(meta), styles["Normal"]), Spacer(1,10)]
    data=[list(df.columns)] + df.astype(str).values.tolist()
    t=Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.25,colors.grey)
    ]))
    story.append(t)
    doc.build(story)
    return buf.getvalue()

# ---------------- UI ----------------
st.set_page_config(page_title="Depreciation Schedule Generator", page_icon="🧮", layout="wide")
st.title("🧮 Depreciation Schedule Generator")

with st.form("depr_form"):
    c1,c2,c3 = st.columns(3)

    with c1:
        asset_name = st.text_input("Asset Name", value="Vehicle ABC 2167")
        asset_type = st.selectbox("Asset Type",
                                  ["Computer & IT","Furniture","Vehicle","Plant & Machinery","Building","Other"], index=2)
        cost = st.number_input("Cost", min_value=0.0, value=5_000_000.0, step=1000.0, format="%.2f")
        salvage = st.number_input("Salvage Value", min_value=0.0, value=500_000.0, step=1000.0, format="%.2f")

    with c2:
        purchase_dt = st.date_input("Purchase Date", value=date.today())
        convention = st.selectbox("Convention",
                                  ["Full month in purchase / None in disposal", "Exact number of days (pro-rata)"])

    with c3:
        dep_method = st.selectbox("Depreciation Method", ["Straight Line (SLM)","Written Down Value (WDV)"])
        rate_pct = st.number_input("Depreciation Rate % (annual)", min_value=0.01, value=20.0, step=0.01)

    submitted = st.form_submit_button("Generate Schedule")

def _yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["Period Start"] = pd.to_datetime(tmp["Period Start"])
    tmp["Year"] = tmp["Period Start"].dt.year.astype(int)
    agg = tmp.groupby("Year", as_index=False).agg(
        Opening_NBV=("Opening NBV", "first"),
        Depreciation=("Depreciation", "sum"),
        Closing_NBV=("Closing NBV", "last"),
    )
    for c in ["Opening_NBV", "Depreciation", "Closing_NBV"]:
        agg[c] = agg[c].round(2)
    agg["Year"] = agg["Year"].astype(str)
    agg = agg.rename(columns={
        "Year": "Year",
        "Opening_NBV": "Opening NBV",
        "Depreciation": "Depreciation",
        "Closing NBV": "Closing NBV",
    })
    return agg

if submitted:
    method_key = 'SLM' if dep_method.startswith('Straight') else 'WDV'
    conv_key = 'full-month' if convention.startswith('Full') else 'exact-days'
    try:
        df = generate_schedule(
            cost=cost,
            salvage=salvage,
            purchase_dt=purchase_dt,
            method=method_key,
            rate_pct=rate_pct,
            convention=conv_key
        )
        st.success("Schedule generated successfully ✅")

        hdr = {
            "Asset Name": asset_name,
            "Asset Type": asset_type,
            "Method": dep_method,
            "Rate %": f"{rate_pct:.2f}%",
            "Cost": f"{cost:,.2f}",
            "Salvage": f"{salvage:,.2f}",
            "Purchase Date": purchase_dt.isoformat(),
            "Convention": convention,
        }

        with st.expander("Asset Details", expanded=True):
            cols = st.columns(4)
            for i,(k,v) in enumerate(hdr.items()):
                with cols[i%4]:
                    st.markdown(f"**{k}**\n\n{v}")

        st.markdown("### Yearly Summary")
        yearly = _yearly_summary(df)
        st.dataframe(yearly, use_container_width=True, hide_index=True)

        with st.expander("Monthly Schedule", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download PDF",
                           data=df_to_pdf_bytes(hdr, df),
                           file_name=f"depreciation_schedule_{asset_name.replace(' ','_')}.pdf",
                           mime="application/pdf")

    except AssertionError as aerr:
        st.error(f"Schedule sanity check failed: {aerr}")
    except Exception as e:
        st.error(f"Error: {e}")
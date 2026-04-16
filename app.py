from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from carbrain_data import (
    build_brand_ranking,
    build_chat_context,
    build_vehicle_metrics,
    get_vehicle_record,
    load_raw_data,
)

load_dotenv(override=True)

st.set_page_config(
    page_title="CarBrain",
    page_icon=":car:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        padding-bottom: 120px;
    }

    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        background-color: #0e1117;
        padding-top: 0.5rem;
        padding-bottom: 0.75rem;
        z-index: 999;
        border-top: 1px solid rgba(250, 250, 250, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CARBRAIN_ASSISTANT_PROMPT = """
Eres CarBrain, un asesor experto en compra de autos seminuevos.

Tu trabajo es ayudar al usuario a decidir si un vehículo le conviene o no,
usando exclusivamente el contexto entregado por la aplicación.

Reglas:
1. Siempre que te salude regresa el saludo y explica para que le puedes ser útil.
2. Responde siempre en español claro, directo y fácil de entender.
3. Habla únicamente del vehículo seleccionado actualmente en el menú izquierdo.
4. Si el usuario menciona otro vehículo distinto, no lo analices; dile que primero lo cambie en el menú desplegable de la izquierda.
5. No inventes datos ni asumas información fuera del contexto.
6. Evita términos técnicos como percentil, cluster, subcluster, silhouette, outlier o score estadístico.
7. Traduce todo a lenguaje práctico para una decisión de compra.
8. Siempre que haya suficiente contexto, intenta cerrar con un veredicto claro:
   - Recomendado
   - Con precaución
   - No recomendado
   
Formato sugerido:
- Resumen breve
- Hallazgos principales
- Qué revisar antes de comprar
- Veredicto final
"""


@st.cache_data(show_spinner=True)
def load_app_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = load_raw_data()
    vehicle_df = build_vehicle_metrics(raw_df)
    brand_df = build_brand_ranking(vehicle_df)
    return vehicle_df, brand_df


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_chat_response(
    user_prompt: str,
    context: str,
    selected_vehicle_text: str,
) -> str:
    client = get_openai_client()
    if client is None:
        return (
            "No encontré la variable OPENAI_API_KEY. "
            "Revisa tu archivo .env y vuelve a intentar."
        )

    model = os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        return (
            "OPENAI_MODEL está vacío. "
            "Define un modelo válido en tu archivo .env."
        )

    try:
        response = client.responses.create(
            model=model,
            instructions=CARBRAIN_ASSISTANT_PROMPT,
            input=(
                f"Vehículo seleccionado en el menú izquierdo: {selected_vehicle_text}\n\n"
                "Debes responder solo sobre ese vehículo. "
                "Si el usuario pregunta por otro auto, indícale que primero lo cambie en el menú de la izquierda.\n\n"
                "Contexto estructurado del análisis de CarBrain:\n"
                f"{context}\n\n"
                "Pregunta del usuario:\n"
                f"{user_prompt}"
            ),
        )

        reply = getattr(response, "output_text", "").strip()
        if not reply:
            return "No pude generar una respuesta útil en este momento."
        return reply

    except Exception as e:
        return f"Error al generar respuesta con OpenAI: {str(e)}"


def get_vehicle_options(vehicle_df: pd.DataFrame, make: str | None = None) -> list[str]:
    if make is None:
        return sorted(vehicle_df["MODELTXT"].dropna().unique().tolist())
    return sorted(
        vehicle_df.loc[vehicle_df["MAKETXT"] == make, "MODELTXT"]
        .dropna()
        .unique()
        .tolist()
    )


def get_year_options(vehicle_df: pd.DataFrame, make: str, model: str) -> list[int]:
    years = (
        vehicle_df.loc[
            (vehicle_df["MAKETXT"] == make) & (vehicle_df["MODELTXT"] == model),
            "YEARTXT",
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return sorted(years, reverse=True)


def make_risk_color(label: str) -> str:
    mapping = {
        "Muy confiable": "#2e8b57",
        "Confiable": "#3cb371",
        "Con precaución": "#d4a017",
        "Riesgoso": "#d2691e",
        "Muy riesgoso": "#b22222",
        "Recomendado": "#2e8b57",
        "No recomendado": "#b22222",
    }
    return mapping.get(label, "#4682b4")


def build_profile_radar(record: pd.Series, title: str) -> go.Figure:
    categories = [
        "Choques reportados",
        "Incendios reportados",
        "Lesiones reportadas",
        "Cantidad de reportes",
        "Severidad general",
    ]

    volume_norm = min(float(record["total_complaints"]) / 500.0, 1.0)
    values = [
        float(record["crash_rate"]),
        float(record["fire_rate"]),
        float(record["injured_rate"]),
        volume_norm,
        float(record["severity_score"]),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=title,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def build_component_bar(record: pd.Series, title: str) -> go.Figure:
    family_data = pd.DataFrame(
        {
            "familia": [
                "Seguridad",
                "Control y manejo",
                "Motor y transmisión",
                "Visibilidad y asistencia",
                "Otros",
            ],
            "peso": [
                float(record.get("seguridad_pasiva", 0.0)),
                float(record.get("seguridad_activa_control", 0.0)),
                float(record.get("propulsion_electrico_termico", 0.0)),
                float(record.get("asistencia_visibilidad", 0.0)),
                float(record.get("otros", 0.0)),
            ],
        }
    )

    fig = px.bar(
        family_data,
        x="familia",
        y="peso",
        title=title,
        text_auto=".2f",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Proporción")
    return fig


def get_prepurchase_checks(record: pd.Series) -> list[str]:
    checks: list[str] = []

    subcluster = str(record.get("subcluster_label", "")).lower()
    top_issue = str(record.get("top_issue", "")).lower()

    if (
        "eléctrico" in subcluster
        or "combustible" in subcluster
        or "fuel" in top_issue
        or "electrical" in top_issue
    ):
        checks.append(
            "Revisar bomba de combustible, líneas, fugas, olor a gasolina y sistema eléctrico."
        )
    if (
        "seguridad" in subcluster
        or "air bags" in top_issue
        or "brakes" in top_issue
        or "steering" in top_issue
    ):
        checks.append(
            "Inspeccionar frenos, dirección, suspensión y sistemas de seguridad."
        )
    if "desgaste" in subcluster or "antigüedad" in subcluster or "suspension" in top_issue:
        checks.append(
            "Revisar desgaste general: suspensión, bujes, llantas, frenos y holguras."
        )
    if "structure" in top_issue or "air bags" in top_issue:
        checks.append(
            "Verificar historial de golpes, alineación estructural y funcionamiento de airbags."
        )
    if "power train" in top_issue or "engine" in top_issue or "propulsion" in top_issue:
        checks.append(
            "Probar transmisión, motor en frío y caliente, ruidos, vibraciones y códigos OBD."
        )

    if float(record.get("crash_rate", 0)) > 0.15:
        checks.append("Pedir inspección mecánica completa y revisar antecedentes de choques.")
    if float(record.get("fire_rate", 0)) > 0.05:
        checks.append("Revisar con especial atención cables, fusibles, conectores y posibles fugas.")
    if float(record.get("injured_rate", 0)) > 0.10:
        checks.append("Priorizar revisión de sistemas críticos de seguridad antes de comprar.")

    if not checks:
        checks.append(
            "Realizar inspección mecánica general, prueba de manejo y escaneo electrónico preventivo."
        )

    return list(dict.fromkeys(checks))[:5]


def init_chat_for_vehicle(selected_vehicle_text: str) -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Hola, soy CarBrain. Ya tengo seleccionado este vehículo: **{selected_vehicle_text}**.\n\n"
                "Puedes preguntarme cosas como:\n"
                "- ¿Es buena compra?\n"
                "- ¿Qué problemas comunes tiene?\n"
                "- ¿Qué debo revisar antes de comprarlo?\n"
                "- ¿Lo recomiendas o no?\n\n"
                "Si quieres evaluar otro auto, cámbialo primero en el menú desplegable de la izquierda."
            ),
        }
    ]


vehicle_df, brand_df = load_app_data()

st.title("CarBrain")
st.caption(
    "Asistente para evaluar autos seminuevos con base en historial de fallas, severidad de reportes y recomendación de compra."
)

with st.sidebar:
    st.header("Selecciona el vehículo a evaluar")

    make_options = sorted(vehicle_df["MAKETXT"].dropna().unique().tolist())
    selected_make = st.selectbox("Marca", make_options, key="main_make")

    model_options = get_vehicle_options(vehicle_df, selected_make)
    selected_model = st.selectbox("Modelo", model_options, key="main_model")

    year_options = get_year_options(vehicle_df, selected_make, selected_model)
    selected_year = st.selectbox("Año", year_options, key="main_year")

    min_complaints = st.slider(
        "Mínimo de reportes para comparativos",
        5,
        200,
        30,
        5,
    )

    st.divider()
    st.info(
        "El chat responderá sobre el vehículo seleccionado aquí. "
        "Para analizar otro auto, primero cámbialo en este menú."
    )

record = get_vehicle_record(vehicle_df, selected_make, selected_model, selected_year)
brand_table = build_brand_ranking(vehicle_df, min_complaints=min_complaints)
selected_vehicle_text = f"{selected_make} {selected_model} {selected_year}"

if "active_vehicle_key" not in st.session_state:
    st.session_state.active_vehicle_key = selected_vehicle_text
    init_chat_for_vehicle(selected_vehicle_text)
elif st.session_state.active_vehicle_key != selected_vehicle_text:
    st.session_state.active_vehicle_key = selected_vehicle_text
    init_chat_for_vehicle(selected_vehicle_text)

st.subheader("Resumen del vehículo")

if record is None:
    st.warning("No se encontró información para la combinación seleccionada.")
else:
    risk_color = make_risk_color(str(record.get("risk_label", "")))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Veredicto", str(record.get("decision", "No disponible")))
    col2.metric("Nivel de atención", f"{float(record['risk_score']):.3f}")
    col3.metric("Total de reportes", int(record["total_complaints"]))
    col4.metric("Falla principal", str(record["top_issue"]))

    comparacion_similares = float(record.get("risk_percentile", 0))
    comparacion_texto = (
        "Menor que la mayoría"
        if comparacion_similares <= 0.33
        else "En línea con otros similares"
        if comparacion_similares <= 0.66
        else "Más delicado que la mayoría"
    )

    st.markdown(
        f"""
        <div style="padding:16px;border-radius:12px;background-color:#111827;border-left:8px solid {risk_color};margin-top:8px;">
            <div style="font-size:22px;font-weight:700;">{selected_make} {selected_model} {selected_year}</div>
            <div style="margin-top:8px;font-size:16px;">
                Recomendación general: <b>{record.get("risk_label", "No disponible")}</b><br>
                Comparación frente a autos similares: <b>{comparacion_texto}</b><br>
                Grupo de vehículos parecidos: <b>{record.get("cluster_label", "No disponible")}</b><br>
                Tipo de fallas más comunes: <b>{record.get("subcluster_label", "No disponible")}</b><br>
                Descripción general: <b>{record.get("cluster_description", "No disponible")}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab1, tab2 = st.tabs(
    [
        "Detalle del vehículo",
        "Comparador",
    ]
)

with tab1:
    if record is None:
        st.warning("No se encontró información para la combinación seleccionada.")
    else:
        c1, c2 = st.columns([1.1, 1], gap="large")

        with c1:
            radar = build_profile_radar(record, "Perfil general del vehículo")
            st.plotly_chart(radar, width="stretch")

        with c2:
            component_bar = build_component_bar(record, "Distribución de tipos de falla")
            st.plotly_chart(component_bar, width="stretch")

        st.subheader("Qué revisar antes de comprar")

        checks = get_prepurchase_checks(record)
        for idx, item in enumerate(checks, start=1):
            st.markdown(f"{idx}. {item}")

        st.subheader("Comparación contra el promedio de su marca")

        brand_avg = (
            vehicle_df.loc[vehicle_df["MAKETXT"] == record["MAKETXT"], "risk_score"].mean()
        )
        compare_df = pd.DataFrame(
            {
                "grupo": ["Vehículo seleccionado", f"Promedio de {record['MAKETXT']}"],
                "valor": [float(record["risk_score"]), float(brand_avg)],
            }
        )
        fig_compare_brand = px.bar(
            compare_df,
            x="grupo",
            y="valor",
            text_auto=".3f",
            title="Qué tan delicado es frente al promedio de su marca",
        )
        fig_compare_brand.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Nivel de atención",
        )
        st.plotly_chart(fig_compare_brand, width="stretch")

with tab2:
    st.subheader("Comparador de dos vehículos")

    comp_col1, comp_col2 = st.columns(2, gap="large")

    with comp_col1:
        comp_make_1 = st.selectbox(
            "Marca vehículo A",
            sorted(vehicle_df["MAKETXT"].dropna().unique().tolist()),
            key="comp_make_1",
        )
        comp_model_1 = st.selectbox(
            "Modelo vehículo A",
            get_vehicle_options(vehicle_df, comp_make_1),
            key="comp_model_1",
        )
        comp_year_1 = st.selectbox(
            "Año vehículo A",
            get_year_options(vehicle_df, comp_make_1, comp_model_1),
            key="comp_year_1",
        )

    with comp_col2:
        comp_make_2 = st.selectbox(
            "Marca vehículo B",
            sorted(vehicle_df["MAKETXT"].dropna().unique().tolist()),
            index=min(1, len(make_options) - 1),
            key="comp_make_2",
        )
        comp_model_2 = st.selectbox(
            "Modelo vehículo B",
            get_vehicle_options(vehicle_df, comp_make_2),
            key="comp_model_2",
        )
        comp_year_2 = st.selectbox(
            "Año vehículo B",
            get_year_options(vehicle_df, comp_make_2, comp_model_2),
            key="comp_year_2",
        )

    record_a = get_vehicle_record(vehicle_df, comp_make_1, comp_model_1, comp_year_1)
    record_b = get_vehicle_record(vehicle_df, comp_make_2, comp_model_2, comp_year_2)

    if record_a is None or record_b is None:
        st.warning("No fue posible encontrar uno de los vehículos seleccionados.")
    else:
        name_a = f"{comp_make_1} {comp_model_1} {comp_year_1}"
        name_b = f"{comp_make_2} {comp_model_2} {comp_year_2}"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("A - Veredicto", str(record_a.get("decision", "No disponible")))
        m2.metric("B - Veredicto", str(record_b.get("decision", "No disponible")))
        m3.metric("A - Nivel de atención", f"{float(record_a['risk_score']):.3f}")
        m4.metric("B - Nivel de atención", f"{float(record_b['risk_score']):.3f}")

        compare_metrics = pd.DataFrame(
            {
                "métrica": [
                    "Nivel de atención",
                    "Comparación frente a similares",
                    "Total de reportes",
                    "Choques reportados",
                    "Incendios reportados",
                    "Lesiones reportadas",
                ],
                name_a: [
                    float(record_a["risk_score"]),
                    float(record_a.get("risk_percentile", 0)),
                    float(record_a["total_complaints"]),
                    float(record_a["crash_rate"]),
                    float(record_a["fire_rate"]),
                    float(record_a["injured_rate"]),
                ],
                name_b: [
                    float(record_b["risk_score"]),
                    float(record_b.get("risk_percentile", 0)),
                    float(record_b["total_complaints"]),
                    float(record_b["crash_rate"]),
                    float(record_b["fire_rate"]),
                    float(record_b["injured_rate"]),
                ],
            }
        )

        compare_long = compare_metrics.melt(
            id_vars="métrica",
            var_name="vehículo",
            value_name="valor",
        )

        fig_comp = px.bar(
            compare_long,
            x="métrica",
            y="valor",
            color="vehículo",
            barmode="group",
            title="Comparación directa",
        )
        st.plotly_chart(fig_comp, width="stretch")

        r1, r2 = st.columns(2, gap="large")
        with r1:
            st.plotly_chart(build_profile_radar(record_a, name_a), width="stretch")
        with r2:
            st.plotly_chart(build_profile_radar(record_b, name_b), width="stretch")

        summary_df = pd.DataFrame(
            {
                "vehículo": [name_a, name_b],
                "risk_score": [float(record_a["risk_score"]), float(record_b["risk_score"])],
                "comparacion": [
                    float(record_a.get("risk_percentile", 0)),
                    float(record_b.get("risk_percentile", 0)),
                ],
                "reportes": [int(record_a["total_complaints"]), int(record_b["total_complaints"])],
                "veredicto": [record_a.get("decision", ""), record_b.get("decision", "")],
                "tipo_falla": [record_a.get("subcluster_label", ""), record_b.get("subcluster_label", "")],
            }
        ).sort_values("risk_score", ascending=True)

        best_vehicle = summary_df.iloc[0]
        st.markdown(
            f"""
            <div style="padding:16px;border-radius:12px;background-color:#0f172a;margin-top:8px;">
                <div style="font-size:18px;font-weight:700;">Lectura rápida del comparador</div>
                <div style="margin-top:8px;">
                    El vehículo que sale mejor parado en esta comparación es <b>{best_vehicle["vehículo"]}</b>.
                    Su veredicto actual es <b>{best_vehicle["veredicto"]}</b> y el patrón principal de fallas reportadas es
                    <b>{best_vehicle["tipo_falla"]}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.subheader("Asistente CarBrain")
st.caption(
    "Haz preguntas sobre el vehículo seleccionado en el menú de la izquierda. "
    "Si quieres evaluar otro auto, primero cámbialo ahí."
)

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input(f"Pregunta por {selected_vehicle_text}...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        if record is None:
            reply = (
                "No tengo datos suficientes del vehículo seleccionado para darte una recomendación. "
                "Prueba con otra combinación de marca, modelo y año."
            )
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            context = build_chat_context(record, brand_table, vehicle_df)
            with st.chat_message("assistant"):
                with st.spinner("Analizando historial y generando recomendación..."):
                    reply = generate_chat_response(
                        user_prompt=prompt,
                        context=context,
                        selected_vehicle_text=selected_vehicle_text,
                    )
                st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
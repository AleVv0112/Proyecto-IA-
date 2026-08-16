"""
Asistente de consultas laborales — Interfaz web con Streamlit
Proyecto IA — ULACIT

Para ejecutar localmente:
    pip install -r requirements.txt
    python -m spacy download es_core_news_sm
    streamlit run app.py
"""

import joblib
import spacy
import streamlit as st

# ── Carga del modelo ──────────────────────────────────────────────────────────
nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])
modelo = joblib.load("modelo_clasificador_nb.joblib")

UMBRAL_CONFIANZA = 0.50

# ── Respuestas legales (Código de Trabajo de Costa Rica) ──────────────────────
RESPUESTAS = {
    "aguinaldo": (
        "📋 **Aguinaldo** — Código de Trabajo Art. 166-168\n\n"
        "El aguinaldo equivale a la **doceava parte** del salario total devengado "
        "entre el 1° de diciembre del año anterior y el 30 de noviembre del año en curso. "
        "Se paga en los **primeros 20 días de diciembre**.\n\n"
        "• Si tiene menos de un año laborado → recibe el proporcional al tiempo trabajado.\n"
        "• Aplica para jornadas completas, parciales y trabajos por horas.\n"
        "• No se le puede rebajar impuesto de renta al aguinaldo."
    ),
    "vacaciones": (
        "📋 **Vacaciones** — Código de Trabajo Art. 153-162\n\n"
        "Corresponden **2 semanas de vacaciones** por cada 50 semanas trabajadas en forma continua.\n\n"
        "• Si termina la relación antes del año → recibe vacaciones proporcionales (1 día por mes).\n"
        "• El patrono puede fijar la fecha, pero las vacaciones deben disfrutarse.\n"
        "• Solo al finalizar el contrato pueden pagarse en dinero."
    ),
    "periodo_prueba": (
        "📋 **Periodo de prueba** — Código de Trabajo Art. 22\n\n"
        "El periodo de prueba puede durar **hasta 3 meses**. Durante este lapso:\n\n"
        "• Cualquiera de las partes puede terminar el contrato sin responsabilidad.\n"
        "• El trabajador mantiene **todos sus derechos**: salario mínimo, CCSS, "
        "aguinaldo proporcional y vacaciones proporcionales.\n"
        "• El tiempo en prueba cuenta para efectos de antigüedad."
    ),
    "rebajos_ley": (
        "📋 **Rebajos de ley** — Código de Trabajo y Ley CCSS\n\n"
        "Las únicas deducciones **obligatorias** son:\n\n"
        "• **Cuota CCSS** ≈ 10.67% del salario bruto (cubre Salud, IVM y Banco Popular).\n"
        "• **Impuesto sobre la renta**, únicamente si el salario supera el mínimo exento "
        "fijado por el Ministerio de Hacienda.\n\n"
        "⚠️ Cualquier otro descuento requiere **autorización escrita** del trabajador."
    ),
    "jornada_horas_extra": (
        "📋 **Jornada y horas extra** — Código de Trabajo Art. 136-145\n\n"
        "• Jornada **diurna**: máx. 8 h/día y 48 h/semana.\n"
        "• Jornada **nocturna**: máx. 6 h/día y 36 h/semana.\n"
        "• Jornada **mixta**: máx. 7 h/día.\n\n"
        "• Las horas extra se pagan con **50% adicional** sobre la hora ordinaria.\n"
        "• Trabajo en **feriados** paga doble.\n"
        "• No puede exigirse más de 12 horas en un mismo día (ordinarias + extras)."
    ),
}

MENSAJE_ESCALAR = (
    "⚠️ **Esta consulta requiere atención especializada.**\n\n"
    "La derivamos a un asesor humano de la organización. "
    "Por favor comunícate directamente con el equipo de orientación laboral.\n\n"
    "_Recuerda: este sistema no sustituye la asesoría legal profesional._"
)

ETIQUETAS_ES = {
    "aguinaldo":          "Aguinaldo",
    "vacaciones":         "Vacaciones",
    "periodo_prueba":     "Periodo de prueba",
    "rebajos_ley":        "Rebajos de ley",
    "jornada_horas_extra":"Jornada / Horas extra",
    "escalar_humano":     "Caso grave → Asesor humano",
}

# ── Funciones de procesamiento ────────────────────────────────────────────────
def limpiar_texto(texto: str) -> str:
    doc = nlp(str(texto).lower())
    tokens = [
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct
        and not t.is_space and len(t.lemma_) > 1
    ]
    return " ".join(tokens)


def clasificar(consulta: str):
    """Clasifica una consulta y devuelve los datos que muestra la interfaz."""
    if not consulta or not consulta.strip():
        return "—", 0.0, "Por favor escribe tu consulta.", ""

    limpio    = limpiar_texto(consulta)
    categoria = modelo.predict([limpio])[0]
    confianza = float(modelo.predict_proba([limpio]).max())

    if categoria == "escalar_humano" or confianza < UMBRAL_CONFIANZA:
        accion   = "⚠️ Derivar a asesor humano"
        respuesta = MENSAJE_ESCALAR
    else:
        accion   = "✅ Respuesta automática"
        respuesta = RESPUESTAS.get(categoria, "")

    etiqueta = ETIQUETAS_ES.get(categoria, categoria)
    return etiqueta, round(confianza, 2), accion, respuesta


# ── Interfaz Streamlit ───────────────────────────────────────────────────────
st.set_page_config(page_title="Asistente Laboral — ULACIT", page_icon="🤝")

st.title("🤝 Asistente de Consultas Laborales")
st.write("Para jóvenes en su primer empleo formal en Costa Rica")
st.info(
    "Escribe tu consulta en lenguaje natural. El sistema la clasifica y "
    "muestra una orientación basada en el Código de Trabajo de Costa Rica."
)
st.warning("Este sistema no sustituye la asesoría legal profesional.")

ejemplos = [
    "cuanto me toca de aguinaldo si entre en agosto",
    "cuantos dias de vacaciones me corresponden",
    "por que me rebajan tanto del salario",
    "me pueden botar en el periodo de prueba",
    "me hacen trabajar 10 horas todos los dias es legal",
    "me despidieron sin pagarme nada y no se que hacer",
]

consulta = st.text_area(
    "Tu consulta",
    placeholder="Ej: cuanto me toca de aguinaldo si entre en agosto...",
    height=120,
)

st.caption("También puedes probar una consulta de ejemplo:")
ejemplo_elegido = st.selectbox("Ejemplos", ["Selecciona un ejemplo"] + ejemplos)

if st.button("Consultar", type="primary"):
    consulta_final = consulta.strip()
    if not consulta_final and ejemplo_elegido != "Selecciona un ejemplo":
        consulta_final = ejemplo_elegido

    if not consulta_final:
        st.error("Por favor escribe una consulta o selecciona un ejemplo.")
    else:
        etiqueta, confianza, accion, respuesta = clasificar(consulta_final)

        col_categoria, col_confianza, col_accion = st.columns(3)
        with col_categoria:
            st.metric("Categoría detectada", etiqueta)
        with col_confianza:
            st.metric("Confianza", f"{confianza:.2f}")
        with col_accion:
            st.metric("Acción", accion)

        st.subheader("Respuesta")
        st.markdown(respuesta)

st.divider()
st.caption(
    "Proyecto: Asistente IA para clasificación de consultas laborales | "
    "Curso: Inteligencia Artificial — ULACIT | "
    "Tecnologías: Python, spaCy, scikit-learn y Streamlit"
)

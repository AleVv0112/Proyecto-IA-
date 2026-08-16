"""
Asistente de consultas laborales — Interfaz web con Streamlit
Proyecto IA — ULACIT

Para ejecutar localmente:
    pip install -r requirements.txt
    python -m spacy download es_core_news_sm
    streamlit run app.py

Necesita una variable de entorno / secreto GEMINI_API_KEY (gratis en
https://aistudio.google.com/apikey) para generar las respuestas con IA.
"""

import os
import re

import google.generativeai as genai
import joblib
import spacy
import streamlit as st

# ── Carga del modelo clasificador ─────────────────────────────────────────────
nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])
modelo = joblib.load("modelo_clasificador_nb.joblib")

UMBRAL_CONFIANZA = 0.50

# ── Configuración de Gemini (genera el texto de las respuestas) ──────────────
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

MENSAJE_FUERA_DOMINIO = (
    "**Esta consulta no parece estar relacionada con temas laborales.**\n\n"
    "Solo puedo ayudarte con dudas sobre aguinaldo, vacaciones, periodo de prueba, "
    "rebajos de ley y jornada/horas extra. Intenta reformular tu pregunta enfocándola "
    "en alguno de esos temas."
)

ETIQUETAS_ES = {
    "aguinaldo":          "Aguinaldo",
    "vacaciones":         "Vacaciones",
    "periodo_prueba":     "Periodo de prueba",
    "rebajos_ley":        "Rebajos de ley",
    "jornada_horas_extra":"Jornada / Horas extra",
    "escalar_humano":     "Caso grave → Asesor humano",
    "no_relacionado":     "Fuera de ámbito laboral",
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


MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def extraer_datos_consulta(consulta: str, categoria: str) -> dict:
    """Extrae datos cuantificables para incorporarlos a la respuesta."""
    datos = {}
    consulta_normalizada = consulta.lower().replace(",", "")

    if categoria == "aguinaldo":
        mes_encontrado = next(
            (numero for nombre, numero in MESES.items() if nombre in consulta_normalizada),
            None,
        )
        if mes_encontrado:
            meses_trabajados = 12 - mes_encontrado if mes_encontrado <= 11 else 0
            datos["mes_ingreso"] = mes_encontrado
            datos["meses_computables_estimados"] = meses_trabajados
            datos["porcentaje_salario_estimado"] = round(meses_trabajados / 12 * 100, 2)

        monto = re.search(
            r"(?:₡|¢|colones?\s*)\s*([\d.]+(?:,\d{1,2})?)(?:\s*mil)?|"
            r"([\d.]+(?:,\d{1,2})?)\s*mil",
            consulta_normalizada,
        )
        if monto:
            monto_texto = next(grupo for grupo in monto.groups() if grupo is not None)
            es_mil = "mil" in monto.group(0)
            monto_texto = monto_texto.replace(".", "").replace(",", ".")
            salario = float(monto_texto)
            if es_mil:
                salario *= 1000
            datos["salario_mensual_mencionado"] = salario
            if "porcentaje_salario_estimado" in datos:
                datos["aguinaldo_estimado"] = round(
                    salario * datos["porcentaje_salario_estimado"] / 100, 2
                )

    return datos


def generar_respuesta_ia(consulta: str, categoria: str) -> str:
    """Usa Gemini para redactar una respuesta personalizada a la consulta."""
    contexto_legal = RESPUESTAS.get(categoria, "")
    datos_extraidos = extraer_datos_consulta(consulta, categoria)
    pide_calculo = bool(
        re.search(
            r"\b(cu[aá]nto|calcular|c[aá]lculo|porcentaje|proporci[oó]n|total|diferencia|"
            r"suma|resta|multiplicar|dividir|d[ií]as|horas|monto|pagar[ií]a|ganar[ií]a)\b",
            consulta.lower(),
        )
    )
    calculo = (
        f"Datos extraídos y cálculo preliminar: {datos_extraidos}\n"
        "Si falta el salario o hay salarios variables, explica que el resultado es solo "
        "una estimación y solicita el dato necesario.\n"
        if datos_extraidos and pide_calculo
        else "La consulta puede requerir cálculo: identifica los números, unidades y "
        "periodo; si hay datos suficientes, realiza la operación.\n"
        if pide_calculo
        else "La consulta no pide un cálculo explícito.\n"
    )
    prompt = (
        "Eres un asistente laboral que orienta a jovenes costarricenses en su primer "
        "empleo formal.\n\n"
        f'Pregunta del usuario: "{consulta}"\n'
        f"Tema detectado: {ETIQUETAS_ES.get(categoria, categoria)}\n\n"
        f"Informacion oficial de referencia (Codigo de Trabajo de Costa Rica):\n{contexto_legal}\n\n"
        f"{calculo}\n"
        "Instrucciones:\n"
        "- Responde en español, de forma clara, cercana y natural, como si conversaras "
        "con la persona.\n"
        "- Personaliza la respuesta a los detalles que haya dado en su pregunta (fechas, "
        "montos, meses trabajados, etc.) si los mencionó.\n"
        "- Si se proporcionó un cálculo preliminar, muéstralo claramente y explica la "
        "fórmula. Para aguinaldo, usa: suma de salarios del periodo aplicable dividida "
        "entre 12; con salario mensual constante, meses computables dividido entre 12 "
        "del salario mensual. No inventes un monto si falta el salario.\n"
        "- Si la pregunta solicita cualquier cálculo sencillo, resuélvelo aunque no sea "
        "aguinaldo: porcentajes, proporciones, sumas, diferencias, días, horas, salario "
        "por hora, pago de horas extra o deducciones. Muestra los datos usados, la fórmula "
        "y el resultado con sus unidades.\n"
        "- Si faltan datos para calcular, no adivines: indica qué dato falta y, cuando sea "
        "útil, muestra la fórmula que la persona puede completar.\n"
        "- Basa la respuesta unicamente en la informacion oficial de referencia; no "
        "inventes articulos, porcentajes ni plazos que no esten ahi.\n"
        "- Cierra siempre aclarando que es orientacion general y no sustituye asesoria "
        "legal profesional."
    )
    try:
        modelo_ia = genai.GenerativeModel(GEMINI_MODEL)
        return modelo_ia.generate_content(prompt).text.strip()
    except Exception:
        # Si falla la llamada a Gemini (sin API key, sin cuota, sin internet...)
        return contexto_legal


def clasificar(consulta: str):
    """Clasifica una consulta y devuelve los datos que muestra la interfaz."""
    if not consulta or not consulta.strip():
        return "—", 0.0, "Por favor escribe tu consulta."

    limpio    = limpiar_texto(consulta)
    categoria = modelo.predict([limpio])[0]
    confianza = float(modelo.predict_proba([limpio]).max())

    if categoria == "no_relacionado":
        respuesta = MENSAJE_FUERA_DOMINIO
    elif categoria == "escalar_humano" or confianza < UMBRAL_CONFIANZA:
        respuesta = MENSAJE_ESCALAR
    else:
        respuesta = generar_respuesta_ia(consulta, categoria)

    etiqueta = ETIQUETAS_ES.get(categoria, categoria)
    return etiqueta, round(confianza, 2), respuesta


# ── Interfaz Streamlit — Chatbot ─────────────────────────────────────────────
st.set_page_config(page_title="Asistente Laboral — ULACIT", page_icon="🤝")

st.title("🤝 Asistente de Consultas Laborales")
st.write("Para jóvenes en su primer empleo formal en Costa Rica")
st.write(
    "Escríbeme tus consultas en lenguaje natural, una por una, como en un chat. "
    "Las clasifico y te muestro una orientación basada en el Código de Trabajo de Costa Rica."
)

ejemplos = [
    "cuanto me toca de aguinaldo si entre en agosto",
    "cuantos dias de vacaciones me corresponden",
    "por que me rebajan tanto del salario",
    "me pueden botar en el periodo de prueba",
    "me hacen trabajar 10 horas todos los dias es legal",
    "me despidieron sin pagarme nada y no se que hacer",
]

with st.sidebar:
    st.subheader("Ejemplos de consultas")
    for ejemplo in ejemplos:
        st.markdown(f"- {ejemplo}")
    st.divider()
    if st.button("🗑️ Borrar conversación"):
        st.session_state.mensajes = []
        st.rerun()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostramos el historial de la conversación
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        detalle = mensaje.get("detalle")
        if detalle:
            col_categoria, col_confianza = st.columns(2)
            with col_categoria:
                st.metric("Categoría detectada", detalle["etiqueta"])
            with col_confianza:
                st.metric("Confianza", f"{detalle['confianza']:.2f}")

consulta = st.chat_input("Escribe tu consulta laboral aquí...")

if consulta:
    st.session_state.mensajes.append({"role": "user", "content": consulta})
    with st.chat_message("user"):
        st.markdown(consulta)

    etiqueta, confianza, respuesta = clasificar(consulta)

    with st.chat_message("assistant"):
        col_categoria, col_confianza = st.columns(2)
        with col_categoria:
            st.metric("Categoría detectada", etiqueta)
        with col_confianza:
            st.metric("Confianza", f"{confianza:.2f}")
        st.markdown(respuesta)

    st.session_state.mensajes.append({
        "role": "assistant",
        "content": respuesta,
        "detalle": {"etiqueta": etiqueta, "confianza": confianza},
    })

st.divider()
st.caption(
    "Proyecto: Asistente IA para clasificación de consultas laborales | "
    "Curso: Inteligencia Artificial — ULACIT | "
    "Tecnologías: Python, spaCy, scikit-learn y Streamlit"
)

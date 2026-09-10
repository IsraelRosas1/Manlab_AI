import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from datetime import date, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

DB_DSN = os.environ["MANLAB_DB_DSN"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# reused across requests instead of opening a new DB connection each time
db_pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=DB_DSN)


@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


FRENTES = {
    "f_intelectual": "Intelectual",
    "f_espiritual": "Espiritual",
    "f_fisico": "Físico",
    "f_economico": "Económico",
    "f_social_atraccion": "Social/Atracción",
}


def get_last_7_days_bitacoras(user_id: str) -> list[dict]:
    """Fetch this user's active enrollment logs from the last 7 days, oldest first."""
    since = date.today() - timedelta(days=7)

    query = """
        SELECT
            rdl.log_date,
            rdl.day_index,
            rdl.f_intelectual,
            rdl.f_espiritual,
            rdl.f_fisico,
            rdl.f_economico,
            rdl.f_social_atraccion,
            rdl.note,
            rdl.is_complete
        FROM reto_daily_logs rdl
        JOIN reto_enrollments re ON re.id = rdl.enrollment_id
        WHERE re.user_id = %s
          AND rdl.log_date >= %s
        ORDER BY rdl.log_date ASC;
    """

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (user_id, since))
            return [dict(row) for row in cur.fetchall()]


def build_frentes_summary(logs: list[dict]) -> str:
    """Turn boolean flags into a plain-text failure count per frente for the prompt."""
    fails = {label: 0 for label in FRENTES.values()}
    for log in logs:
        for col, label in FRENTES.items():
            if log[col] is False:
                fails[label] += 1
    return ", ".join(f"{label}: {count} días fallados" for label, count in fails.items())


def build_bitacoras_text(logs: list[dict]) -> str:
    lines = []
    for log in logs:
        note = log["note"] or "(sin nota)"
        lines.append(f"Día {log['day_index']} ({log['log_date']}): {note}")
    return "\n".join(lines) if lines else "Sin registros en los últimos 7 días."


def generate_veredicto(user_id: str) -> str:
    logs = get_last_7_days_bitacoras(user_id)

    frentes_summary = build_frentes_summary(logs)
    bitacoras_text = build_bitacoras_text(logs)

    system_prompt = (
        """Eres Izahi Santana de ManLab Project. Hablas en su voz: directa, confrontativa,
        digna, registro mexicano informal pero serio. NO eres autoayuda. NO validas. NO
        consuelas. Eres el espejo brutal del estándar. Hablas espanol mexicano, ocasionalmente dices carnal y cabron.

        Tu tarea: leer la bitácora del Reto 100 de 100 del hombre y darle
        un VEREDICTO corto (10 líneas). Conecta los frentes que está fallando con la
        doctrina todos los frentes se afectan entre sí (cuando cae el
        físico, arrastra al económico y al social; cuando cae el espiritual, se nubla todo).
        Nombra el eslabón débil sin rodeos. Recuerda la doctrina INEVITABILIDAD cuando aplique.

        REGLAS DE VOZ Y MARCA (obligatorias):
        - El Reto NO es sobre confianza, hábitos ni disciplina por estado de ánimo. Es sobre
        PROGRAMAR LA MENTE: que la mente no te diga qué hacer, tú le digas a la mente.
        - Nunca uses la palabra "marco" ni "frame": usa "postura".
        - Nunca uses "seducción"/"seducir" en este contexto: usa atracción, magnetismo,
        presencia, postura.
        - "Sé ese tipo de hombre" SOLO puede aparecer como CIERRE doctrinal, jamás como
        apertura ni en medio. Úsalo con moderación, no siempre.
        - Frases firmadas de Master que puedes usar tal cual:
        "No necesito sentirme bien para hacer las cosas; hago las cosas para sentirme bien."
        "Las creencias se rompen con evidencias."
        "Tú no eres tu mente, tu mente es tuya."
        -no hables con vinetas * ni emojis
        - Si lleva varios días fallando el mismo frente, sé más duro, no más suave.
        - Cita fechas específicas de la bitácora, nunca generalices sin
        evidencia. Si el usuario dice que hizo algo pero la bandera del frente
        correspondiente está en false, señala esa contradicción explícitamente
        (ej: "dices que estudiaste pero tu frente intelectual quedó marcado como
        incompleto").
        - Detecta patrones de "hacer cosas" sin "cumplir disciplina": actividades
        sueltas, sin estructura, sin meta ni fecha de entrega, cuentan como
        distracción aunque suenen productivas."""
                )

    user_prompt = (
        f"Resumen de fallos por frente (últimos 7 días): {frentes_summary}\n\n"
        f"Bitácoras diarias:\n{bitacoras_text}\n\n"
        "Da el veredicto de Master: conecta los frentes que está fallando y ciérralo con "
        "una acción concreta para mañana."
    )

    openai_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    response = openai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature= 0.7,
        max_tokens= 600,
    )

    return response.choices[0].message.content


app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.manlabproject.com",
        "http://localhost:5173",
        "http://10.0.0.60:5173",
        "http://35.16.106.252:5173",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/veredicto/{user_id}")
def veredicto(user_id: str):
    try:
        return {"veredicto": generate_veredicto(user_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

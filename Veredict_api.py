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
    """Eres "Master" Izahi Santana de Manlab Project, el mentor del Reto Manlab. Le hablas directo al usuario, sin rodeos,
    en español coloquial mexicano. Usas "hermano" "carnal" o "cabrón" quema, con naturalidad,
    nunca de forma forzada ni en cada frase. No suenas como un coach genérico de
    autoayuda: hablas como alguien que ya vivió esto y no tolera excusas.

    Se te entrega un JSON con la bitácora de los últimos 7 días del usuario. Cada
    entrada tiene: logDate, dayIndex, cinco banderas booleanas por frente
    (fIntelectual, fEspiritual, fFisico, fEconomico, fSocialAtraccion), el texto
    libre "bitacora" que escribió el usuario ese día, e isComplete (si cumplió las
    5 disciplinas al 100%).

    Tu tarea es dar un VEREDICTO, no un resumen. Para eso:

    1. Cita días y fechas específicos de la bitácora en formato mes y dia, nunca generalices sin
    evidencia. Si el usuario dice que hizo algo pero la bandera del frente
    correspondiente está en false, señala esa contradicción explícitamente
    (ej: "dices que estudiaste pero tu frente intelectual quedó marcado como
    incompleto").
    2. Detecta patrones de "hacer cosas" sin "cumplir disciplina": actividades
    sueltas, sin estructura, sin meta ni fecha de entrega, cuentan como
    distracción aunque suenen productivas.
    3. Señala entradas vacías, genéricas o placeholder (como "string" o bitácoras
    de una sola línea sin sustancia) como falta de claridad del usuario, no las
    ignores.
    4. Identifica el frente más débil de la semana (el que más veces aparece en
    false) y conecta cómo ese frente débil está saboteando o distorsionando los
    demás frentes (el "circuito cerrado": ej. falta de sueño -> bajo rendimiento
    físico -> procrastinación económica).
    5. Si el usuario da contexto extra (racha actual, día de la semana cumplido al
    100%, identidad declarada), úsalo para reforzar el veredicto, pero solo si
    viene en el mensaje; no inventes cifras que no te dieron.
    6. Cierra siempre con una exigencia concreta y accionable: una meta con fecha,
    una hora fija, una sola prioridad a la vez. Nunca cierres con consejos
    genéricos tipo "sigue esforzándote" o "tú puedes".

    7.Si el usuario se desvia del tema, redirige la conversacion.
     
    8.Si el usuario hace algo bien, hazlo notar para que lo vuelva a hacer, y da 
    Formato: párrafos cortos, tono de conversación directa (como si fuera un
    mensaje de voz transcrito), sin viñetas ni listas numeradas, sin emojis, sin
    encabezados. No repitas la bitácora completa, solo cita lo relevante para el
    punto que estás haciendo.

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
    - Si lleva varios días fallando el mismo frente, sé más duro, no más suave.
    - Cierra SIEMPRE con: Honos · Probitas · Perfectio"""
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
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature= 0.7,
        max_tokens= 1200,
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

import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from uuid import UUID

from fastapi import FastAPI, HTTPException, Header, Query

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


def get_bitacoras(
    user_id: str,
    enrollment_id: UUID,
    period_start: date,
    period_end: date
) -> list[dict]:
    query = """
        SELECT
            rdl.log_date,
            rdl.f_intelectual,
            rdl.f_espiritual,
            rdl.f_fisico,
            rdl.f_economico,
            rdl.f_social_atraccion,
            rdl.note,
            rdl.is_complete,
            re.signature AS reto_signature,
            u.intelectual_goal,
            u.espiritual_goal,
            u.fisico_goal,
            u.economico_goal,
            u.social_atraccion_goal
        FROM reto_daily_logs rdl
        JOIN reto_enrollments re
            ON re.id = rdl.enrollment_id
        JOIN users u
            ON u.id = re.user_id
        WHERE re.id = %s
          AND re.user_id = %s
          AND re.status = 'active'
          AND rdl.log_date >= %s
          AND rdl.log_date <= %s
        ORDER BY rdl.log_date ASC;
    """

    with get_db_connection() as conn:
        with conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(
                query,
                (
                    str(enrollment_id),
                    user_id,
                    period_start,
                    period_end
                )
            )
            return [dict(row) for row in cur.fetchall()]


def build_frentes_summary(logs: list[dict]) -> str:
    fails = {label: 0 for label in FRENTES.values()}

    for log in logs:
        for column, label in FRENTES.items():
            if log[column] is False:
                fails[label] += 1

    return ", ".join(
        f"{label}: {count} días fallados"
        for label, count in fails.items()
    )


def build_goals_text(logs: list[dict]) -> str:
    if not logs:
        return "Sin metas disponibles para este usuario."

    first_log = logs[0]
    goals = {
        "Intelectual": first_log["intelectual_goal"],
        "Espiritual": first_log["espiritual_goal"],
        "Físico": first_log["fisico_goal"],
        "Económico": first_log["economico_goal"],
        "Social/Atracción": first_log["social_atraccion_goal"],
    }

    return "\n".join(
        f"{frente}: {goal or '(sin meta definida)'}"
        for frente, goal in goals.items()
    )


def build_bitacoras_text(logs: list[dict]) -> str:
    lines = []

    for log in logs:
        note = log["note"] or "(sin nota)"
        lines.append(
            f"Día ({log['log_date']}): {note}"
        )

    return "\n".join(lines) if lines else "Sin registros en el periodo solicitado."


def generate_veredicto(
    user_id: str,
    enrollment_id: UUID,
    period_start: date,
    period_end: date
) -> str:
    logs = get_bitacoras(
        user_id,
        enrollment_id,
        period_start,
        period_end
    )

    frentes_summary = build_frentes_summary(logs)
    goals_text = build_goals_text(logs)
    reto_signature = (
        logs[0]["reto_signature"]
        if logs and logs[0]["reto_signature"]
        else "Sin contrato personal registrado."
    )
    bitacoras_text = build_bitacoras_text(logs)
    bitacoras_text = (
        f"Contrato personal del usuario:\n{reto_signature}\n\n"
        f"Metas minimas de disciplina diaria del usuario:\n{goals_text}\n\n"
        f"{bitacoras_text}"
    )

    system_prompt = (
        """Eres Izahi Santana de ManLab Project. Hablas en su voz: directa, confrontativa,
    digna, registro mexicano informal pero serio. NO eres autoayuda. NO validas. NO
    consuelas. Eres el espejo brutal del estándar. Hablas en Español mexicano, nunca Ingles.
    No dices 'false' di, incumplido.

    Tu tarea: leer la bitácora diaria del hombre de sus 5 disciplinasy darle
    un VEREDICTO. Conecta los frentes que está fallando con la
    doctrina todos los frentes se afectan entre sí (cuando cae el
    físico, arrastra al económico y al social; cuando cae el espiritual, se nubla todo).
    Nombra el eslabón débil sin rodeos. Recuerda la doctrina INEVITABILIDAD cuando aplique.
    El usuario debe completar al menos sus metas minimas.Lee el 'Contrato personal del usuario'
    y guia al usuario a cumplir con su contrato personal hacia el tipo de hombre que quiere ser,
    no aceptes mediocridad.

    1. Cita día y mes específicos de la bitácora, nunca generalices sin
    evidencia. Si el usuario dice que hizo algo pero la bandera del frente
    correspondiente está en false, señala esa contradicción explícitamente
    (ej: "dices que estudiaste pero tu frente intelectual quedó marcado como
    incompleto").
    2. Detecta patrones de "hacer cosas" sin "cumplir disciplina" o "cumplir disciplina" sin estructure: 
    actividades sueltas, sin estructura, sin meta ni fecha de entrega, cuentan como
    distracción aunque suenen productivas, promueve enfoque.
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
    Formato: párrafos cortos, tono de conversación directa, sin viñetas ni listas numeradas, sin emojis, sin
    encabezados. No repitas la bitácora completa, solo cita lo relevante para el
    punto que estás haciendo.

    9.Manlab Project es experto en Seduccion, Comunicacion, Masculinidad y Exito. Si las disciplinas del usuario
    necesitan aprender de esto di que cheque los cursos disponibles y hable con el clon 'Master Santana' para una mejor ayuda.
    
    10. VEHÍCULO ÚNICO POR FRENTE (análisis semanal, no solo diario): revisa cada 
    frente a nivel de la semana completa, no solo día por día. Si el usuario salta 
    entre actividades distintas sin un motor claro (ej: un día SEO, otro ventas, otro 
    suscripciones, otro aplicar a empleo), señala esto como dispersión estratégica, 
    aunque cada actividad individual suene productiva. Pregúntale directamente cuál 
    es su vehículo principal en ese frente. Dispersión sin un motor claro es apostar, 
    no construir. Esta misma lógica aplica a cualquiera de los 5 frentes, no solo el 
    económico.

    11. NO RELATIVIZAR CON COMPASIÓN: puedes reconocer contexto difícil (presión 
    financiera, cansancio) sin absolver el patrón. Reconocer contexto no es lo mismo 
    que decir "lo respeto" o "no es fracaso". Nombra el contexto y sigue exigiendo 
    enfoque en la misma frase — nunca sueltes la exigencia solo porque el contexto 
    da lástima.

    12. CUMPLIMIENTO LITERAL DE LA META MÍNIMA: la meta mínima del contrato es 
    específica y literal, no un espíritu general. Si la meta dice "estudiar 20 minutos 
    un módulo del curso de Udemy de APIs en C#", solo cuenta avanzar exactamente en 
    ESE curso. Trabajar en su propia app, leer sobre arquitectura, o programar otra 
    cosa relacionada NO cumple la meta, sin importar cuántas horas le dedicó o qué 
    tan productivo o impresionante suene. Señala esto explícitamente cuando ocurra: 
    nombra la meta literal, nombra lo que el usuario hizo en su lugar, y deja claro 
    que son cosas distintas aunque estén en la misma categoría. Esto aplica a los 5 
    frentes: la meta mínima nunca se cumple por "algo parecido" o "algo mejor", solo 
    se cumple haciendo exactamente lo que el contrato dice. Sustituir la meta por 
    otra actividad, sin importar su calidad, es la mente decidiendo por el usuario, 
    no el usuario programando a la mente.
    
    12b. APLICACIÓN SEMANAL DE LA REGLA 12: antes de escribir el veredicto, revisa 
    CADA día de la semana contra la meta mínima literal de cada frente, no solo 
    el día más obvio. Si el patrón se repite varios días o toda la semana, dilo 
    explícitamente como patrón acumulado ("llevas ocho días sin tocar tu meta real 
    de intelectual, no es un día suelto"), no como un solo ejemplo aislado. Antes 
    de elegir qué día usar como acierto de apertura, verifica que ese día también 
    cumpla la meta mínima literal — nunca abras con un día que tú mismo 
    descalificarías más adelante en el mismo veredicto.

    REGLAS DE VOZ Y MARCA (obligatorias):
    - La bitacora es sobre confianza, hábitos ni disciplina por estado de ánimo. Es sobre
    PROGRAMAR LA MENTE: que la mente no te diga qué hacer, tú le digas a la mente.
    - Nunca uses la palabra "marco" ni "frame": usa "postura".
    - "Sé ese tipo de hombre" SOLO puede aparecer como CIERRE doctrinal, jamás como
    apertura ni en medio. Úsalo con moderación, no siempre.
    - Frases firmadas de Master que puedes usar tal cual:
    "No necesito sentirme bien para hacer las cosas; hago las cosas para sentirme bien."
    "Las creencias se rompen con evidencias."
    "Tú no eres tu mente, tu mente es tuya."
    - Si lleva varios días fallando el mismo frente, sé más duro, no más suave.

    ESTRUCTURA (sin usar encabezados ni viñetas): abre nombrando UN acierto real 
    con evidencia específica (día y cita textual). Después nombra el eslabón débil 
    con evidencia. Cierra con una sola exigencia concreta.

    CIERRE: termina siempre con una pregunta retórica directa que confronte al 
    usuario — no solo una orden. Ejemplo de tono: "¿vas a cerrar eso o seguimos 
    con lo mismo?"

    LONGITUD: Máximo 500 palabras"""
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
        max_tokens= 1200,
    )

    return response.choices[0].message.content


app = FastAPI()

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


INTERNAL_API_KEY = os.getenv("VERDICT_INTERNAL_API_KEY")


@app.get("/veredicto/{user_id}")
def veredicto(
    user_id: str,
    enrollment_id: UUID = Query(...),
    period_start: date = Query(..., alias="from"),
    period_end: date = Query(..., alias="to"),
    x_internal_api_key: str | None = Header(default=None)
):
    if INTERNAL_API_KEY:
        if x_internal_api_key != INTERNAL_API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid internal API key"
            )

    today = date.today()

    if period_start > period_end:
        raise HTTPException(
            status_code=400,
            detail="The start date must be before the end date"
        )

    if period_end > today:
        raise HTTPException(
            status_code=400,
            detail="Future dates are not allowed"
        )

    requested_days = (period_end - period_start).days + 1

    if requested_days > 30:
        raise HTTPException(
            status_code=400,
            detail="The maximum period is 30 days"
        )

    try:
        result = generate_veredicto(
            user_id=user_id,
            enrollment_id=enrollment_id,
            period_start=period_start,
            period_end=period_end
        )

        return {
            "veredicto": result
        }

    except HTTPException:
        raise

    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=str(exception)
        )

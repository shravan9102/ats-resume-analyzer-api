
import os
import json
import time
import logging
from typing import Dict, Any, Optional

from groq import (
    Groq,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

logger = logging.getLogger("ats_resume_scorer")

from backend.core.config import GROQ_MODEL, GROQ_TIMEOUT_SECONDS

MAX_RESUME_CHARS = 12000
MAX_JD_CHARS = 8000

_client: Optional[Groq] = None


def _get_client() -> Groq:
    """Create the Groq client when needed."""
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Check the backend .env file."
            )

        _client = Groq(
            api_key=api_key,
            timeout=GROQ_TIMEOUT_SECONDS,
            max_retries=1,
        )

    return _client


def _call_groq(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1200,
) -> str:
    """Call Groq with bounded waiting and timing logs."""
    started = time.perf_counter()
    logger.info("Starting Groq API request using model %s.", GROQ_MODEL)

    try:
        kwargs = dict(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            # Reasoning models (gpt-oss) spend completion tokens on hidden
            # reasoning; a small budget yields an EMPTY answer. Leave headroom.
            max_tokens=max(max_tokens, 4000),
            response_format={"type": "json_object"},
        )
        if "gpt-oss" in GROQ_MODEL:
            kwargs["reasoning_effort"] = "low"

        response = client.chat.completions.create(**kwargs)

        if not response.choices:
            raise ValueError("Groq returned no response choices.")

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise ValueError("Groq returned an empty response.")

        logger.info(
            "Groq API request completed in %.2f seconds.",
            time.perf_counter() - started,
        )
        return content.strip()

    except APITimeoutError:
        logger.exception(
            "Groq API timed out after %.2f seconds.",
            time.perf_counter() - started,
        )
        raise

    except APIConnectionError:
        logger.exception(
            "Could not connect to Groq after %.2f seconds.",
            time.perf_counter() - started,
        )
        raise

    except APIStatusError:
        logger.exception("Groq returned an HTTP API error.")
        raise

    except Exception:
        logger.exception("Unexpected error during Groq API request.")
        raise


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from the model response."""
    if not text or not text.strip():
        raise ValueError("Groq returned an empty response.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        raise ValueError("Expected a JSON object.")
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("No valid JSON object found in Groq response.")

    result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("The extracted JSON must be an object.")

    return result


def _normalize_list_fields(
    result: Dict[str, Any],
    fields: list,
) -> None:
    """Ensure requested fields contain lists."""
    for field in fields:
        value = result.get(field)

        if value is None:
            result[field] = []
        elif isinstance(value, list):
            continue
        elif isinstance(value, str):
            result[field] = [value] if value.strip() else []
        else:
            result[field] = []


def _validate_resume_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize parsed resume data for downstream functions."""
    defaults = {
        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "summary": "",
        "professional_summary": "",
        "skills": [],
        "keywords": [],
        "action_verbs": [],
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "linkedin": "",
        "github": "",
        "portfolio": "",
    }

    for field, default in defaults.items():
        if field not in result or result[field] is None:
            result[field] = default

    _normalize_list_fields(
        result,
        [
            "skills",
            "keywords",
            "action_verbs",
            "education",
            "experience",
            "projects",
            "certifications",
        ],
    )

    for field in [
        "name",
        "email",
        "phone",
        "location",
        "summary",
        "professional_summary",
        "linkedin",
        "github",
        "portfolio",
    ]:
        if not isinstance(result[field], str):
            result[field] = str(result[field])

    # Keep the summary fields compatible with existing code.
    if not result["professional_summary"] and result["summary"]:
        result["professional_summary"] = result["summary"]

    return result


def parse_resume(resume_text: str) -> Dict[str, Any]:
    """Parse resume text into structured JSON using Groq."""
    if not resume_text or not resume_text.strip():
        raise ValueError(
            "No resume text was extracted. "
            "Check whether the PDF contains selectable text."
        )

    resume_text = resume_text[:MAX_RESUME_CHARS]

    system_prompt = """
You are an expert resume parser for an ATS Resume Scorer.

Return ONLY one valid JSON object. Do not use Markdown.
Do not invent missing information.

Use exactly this structure:
{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "summary": "",
  "professional_summary": "",
  "linkedin": "",
  "github": "",
  "portfolio": "",
  "skills": [],
  "keywords": [],
  "action_verbs": [],
  "education": [],
  "experience": [],
  "projects": [],
  "certifications": []
}

Rules:
- skills, keywords and action_verbs must be arrays of strings.
- education, experience, projects and certifications must be arrays.
- Experience items may include duration_months as an integer when supported
  by the resume. Do not guess dates or durations.
- Use empty strings or arrays when information is unavailable.
"""

    user_prompt = (
        "Parse the following resume into the required JSON structure:\n\n"
        f"{resume_text}"
    )

    client = _get_client()
    last_error = None

    # One retry only for malformed JSON, not for API/network errors.
    for attempt in range(1, 3):
        try:
            prompt = user_prompt

            if attempt == 2:
                logger.warning(
                    "Retrying resume parsing once due to invalid JSON."
                )
                prompt += (
                    "\n\nReturn ONLY a valid JSON object. "
                    "Do not include Markdown or explanations."
                )

            raw_response = _call_groq(
                client=client,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=1200,
            )

            result = _extract_json(raw_response)
            result = _validate_resume_result(result)

            logger.info("Resume parsing completed successfully.")
            return result

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Resume parsing attempt %d failed: %s",
                attempt,
                exc,
            )

    raise ValueError(
        "Could not parse valid resume JSON after two attempts: "
        f"{last_error}"
    )


def parse_job_description(job_description: str) -> Dict[str, Any]:
    """Extract skills and requirements from a job description."""
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    job_description = job_description[:MAX_JD_CHARS]

    system_prompt = """
You are an expert ATS job-description analyzer.

Return ONLY one valid JSON object. Do not use Markdown.
Do not invent requirements.

Use this structure:
{
  "job_title": "",
  "required_skills": [],
  "preferred_skills": [],
  "keywords": [],
  "responsibilities": [],
  "qualifications": [],
  "experience_required": ""
}

All list fields must be JSON arrays.
Use empty strings or arrays when information is unavailable.
"""

    user_prompt = (
        "Analyze this job description and return the required JSON:\n\n"
        f"{job_description}"
    )

    client = _get_client()
    last_error = None

    for attempt in range(1, 3):
        try:
            prompt = user_prompt

            if attempt == 2:
                logger.warning(
                    "Retrying JD parsing once due to invalid JSON."
                )
                prompt += (
                    "\n\nReturn ONLY valid JSON. "
                    "Do not include Markdown or explanations."
                )

            raw_response = _call_groq(
                client=client,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_tokens=1000,
            )

            result = _extract_json(raw_response)

            defaults = {
                "job_title": "",
                "required_skills": [],
                "preferred_skills": [],
                "keywords": [],
                "responsibilities": [],
                "qualifications": [],
                "experience_required": "",
            }

            for field, default in defaults.items():
                if field not in result or result[field] is None:
                    result[field] = default

            _normalize_list_fields(
                result,
                [
                    "required_skills",
                    "preferred_skills",
                    "keywords",
                    "responsibilities",
                    "qualifications",
                ],
            )

            for field in ["job_title", "experience_required"]:
                if not isinstance(result[field], str):
                    result[field] = str(result[field])

            logger.info("Job description parsing completed successfully.")
            return result

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "JD parsing attempt %d failed: %s",
                attempt,
                exc,
            )

    raise ValueError(
        "Could not parse valid job-description JSON after two attempts: "
        f"{last_error}"
    )

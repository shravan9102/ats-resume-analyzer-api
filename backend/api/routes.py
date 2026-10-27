
import asyncio
import logging
import time
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from starlette.concurrency import run_in_threadpool

from backend.api.auth import get_current_user
from backend.core.config import (
    ANALYSIS_TIMEOUT_SECONDS,
    HISTORY_SAVE_TIMEOUT_SECONDS,
    MAX_FILE_SIZE_BYTES,
    PARSE_TIMEOUT_SECONDS,
)
from backend.utils.file_utils import FileParsingError, FileValidationError
from backend.models.schemas import (
    AnalysisResponse,
    ComponentScores,
    JDComparison,
    SkillValidationDetails,
)

logger = logging.getLogger("ats_resume_scorer")

router = APIRouter(prefix="/api/v1", tags=["Analysis"])

MAX_RESUME_SIZE = MAX_FILE_SIZE_BYTES


def _status_for_analysis_error(exc: Exception) -> int:
    name = type(exc).__name__
    if name in {"APITimeoutError"}:
        return 504
    if name in {"APIConnectionError", "AuthenticationError", "RateLimitError", "APIStatusError"}:
        return 502
    if isinstance(exc, ValueError):
        return 422
    return 500


def _public_error_message(exc: Exception) -> str:
    """User-facing message that never includes secrets or stack traces."""
    name = type(exc).__name__
    if name == "APITimeoutError":
        return "The AI service (Groq) timed out. Please try again."
    if name == "APIConnectionError":
        return "Could not reach the AI service (Groq). Check your internet connection."
    if name == "AuthenticationError":
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file."
    if name == "RateLimitError":
        return "The AI service rate limit was hit. Wait a moment and retry."
    if isinstance(exc, ValueError):
        return f"Analysis could not be completed: {exc}"
    return f"Analysis failed ({name}). See the backend log for details."


async def _save_history_safely(user_id: str, filename: str, result: dict) -> None:
    try:
        from backend.database.supabase_db import save_analysis

        await asyncio.wait_for(
            save_analysis(user_id, filename, result),
            timeout=HISTORY_SAVE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("History save timed out for %s (analysis was still returned).", filename)
    except Exception:
        logger.exception("History save failed for %s (analysis was still returned).", filename)


@router.post("/analyze-resume", response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    background_tasks: BackgroundTasks,
    resume: UploadFile = File(
        ...,
        description="Resume file - PDF, DOCX or TXT, max 5 MB",
    ),
    job_description: str = Form(
        "",
        description="Job description text (optional)",
    ),
    user_id: str = Depends(get_current_user),
):
    request_start = time.perf_counter()
    filename = resume.filename or "resume"

    logger.info("Resume analysis request received: %s", filename)

    # 1. Verify NLP models.
    nlp = getattr(request.app.state, "nlp", None)
    embedder = getattr(request.app.state, "embedder", None)

    if nlp is None or embedder is None:
        logger.error("NLP or embedding model is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Analysis models are not ready. Restart the backend and retry.",
        )

    # 2. Read and validate the uploaded file.
    try:
        logger.info("STAGE read: reading upload bytes (%s)", filename)
        file_bytes = await resume.read(MAX_RESUME_SIZE + 1)
        logger.info("STAGE read done: %d bytes", len(file_bytes))

        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded resume is empty.")

        if len(file_bytes) > MAX_RESUME_SIZE:
            raise HTTPException(status_code=413, detail="Resume file exceeds the 5 MB limit.")

        # Imported at startup (see main.py); this is a cheap cached lookup.
        from backend.services.resume_parser import parse_resume_file

        parse_start = time.perf_counter()
        logger.info("STAGE parse: extracting text (timeout %ss)", PARSE_TIMEOUT_SECONDS)

        resume_text, metadata = await asyncio.wait_for(
            run_in_threadpool(parse_resume_file, file_bytes, filename),
            timeout=PARSE_TIMEOUT_SECONDS,
        )

        logger.info(
            "Resume parsed: %s | type=%s | characters=%d | duration=%.2fs",
            filename,
            metadata.get("file_type"),
            len(resume_text or ""),
            time.perf_counter() - parse_start,
        )

        if not resume_text or not resume_text.strip():
            raise HTTPException(status_code=422, detail="No readable text was found in the resume.")

    except HTTPException:
        raise

    except asyncio.TimeoutError as exc:
        logger.error("Resume text extraction exceeded %ss: %s", PARSE_TIMEOUT_SECONDS, filename)
        raise HTTPException(
            status_code=504,
            detail=f"Reading the resume file took longer than {PARSE_TIMEOUT_SECONDS}s. "
                   "The file may be malformed; try exporting it to a fresh PDF.",
        ) from exc

    except FileValidationError as exc:
        logger.warning("Upload rejected (%s): %s", filename, exc.message)
        raise HTTPException(status_code=400, detail=exc.message) from exc

    except FileParsingError as exc:
        logger.warning("Resume could not be parsed (%s): %s", filename, exc.message)
        raise HTTPException(status_code=422, detail=exc.message) from exc

    except Exception as exc:
        logger.exception("Resume parsing failed unexpectedly: %s", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while reading the resume: {type(exc).__name__}",
        ) from exc

    finally:
        await resume.close()

    # 3. Run the analysis with a diagnostic timeout.
    try:
        from backend.services.resume_analyzer import analyze_full_resume

        analysis_start = time.perf_counter()
        logger.info("Starting full resume analysis: %s", filename)

        result = await asyncio.wait_for(
            run_in_threadpool(
                analyze_full_resume,
                resume_text=resume_text,
                nlp=nlp,
                embedder=embedder,
                job_description=job_description,
            ),
            timeout=ANALYSIS_TIMEOUT_SECONDS,
        )

        logger.info(
            "Resume analysis completed: %s | duration=%.2fs",
            filename,
            time.perf_counter() - analysis_start,
        )

    except asyncio.TimeoutError as exc:
        logger.error(
            "Resume analysis exceeded %s seconds: %s",
            ANALYSIS_TIMEOUT_SECONDS,
            filename,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Resume analysis exceeded {ANALYSIS_TIMEOUT_SECONDS} seconds. "
                "Check the backend logs (look for the last STEP line) to see "
                "which NLP, embedding or Groq step was slow."
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Resume analysis pipeline failed: %s", filename)
        raise HTTPException(
            status_code=_status_for_analysis_error(exc),
            detail=_public_error_message(exc),
        ) from exc

    if not isinstance(result, dict):
        logger.error("Analyzer returned an unexpected result type.")
        raise HTTPException(
            status_code=500,
            detail="The analysis service returned an invalid result.",
        )

    # 4. Prepare job-description comparison.
    jd_comparison_result = None
    jd_data = result.get("jd_comparison")

    if isinstance(jd_data, dict):
        try:
            jd_comparison_result = JDComparison(
                match_percentage=round(
                    float(jd_data.get("match_percentage", 0.0)), 1
                ),
                semantic_similarity=round(
                    float(jd_data.get("semantic_similarity", 0.0)), 3
                ),
                matched_keywords=jd_data.get("matched_keywords", [])[:20],
                missing_keywords=jd_data.get("missing_keywords", [])[:15],
                skills_gap=jd_data.get("skills_gap", [])[:10],
            )
        except (TypeError, ValueError) as exc:
            logger.exception("Invalid JD comparison result.")
            raise HTTPException(
                status_code=500,
                detail="Could not process job-description comparison.",
            ) from exc

    # 5. Normalize skill-validation details.
    svd_raw = result.get("skill_validation_details") or {}

    try:
        skill_val_details = SkillValidationDetails(
            validated=svd_raw.get("validated", []),
            unvalidated=svd_raw.get("unvalidated", []),
            total=svd_raw.get("total", 0),
            validated_count=svd_raw.get("validated_count", 0),
            validation_pct=svd_raw.get("validation_pct", 0.0),
        )
    except Exception as exc:
        logger.exception("Could not construct skill-validation details.")
        raise HTTPException(
            status_code=500,
            detail="Could not process skill-validation results.",
        ) from exc

    # 6. Build the API response.
    try:
        response = AnalysisResponse(
            ATS_score=result["ats_score"],
            ats_score=result["ats_score"],
            component_scores=ComponentScores(
                **result["component_scores"]
            ),
            issues_summary=result["issues_summary"],
            detailed_feedback=result.get("detailed_feedback", []),
            jd_match_analysis=jd_comparison_result,
            skill_validation_details=skill_val_details,
            keyword_match=(
                jd_comparison_result.match_percentage
                if jd_comparison_result
                else 0.0
            ),
            missing_keywords=result.get("missing_keywords", []),
            matched_keywords=result.get("matched_keywords", []),
            skills=list(result.get("skills", [])[:20]),
            jd_comparison=jd_comparison_result,
            interpretation=result.get("interpretation", ""),
            strengths=list(result.get("strengths", []) or []),
            suggestions=list(result.get("suggestions", []) or []),
            critical_issues=list(result.get("critical_issues", []) or []),
        )
    except Exception as exc:
        logger.exception("Failed to construct analysis response.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not build analysis response: {exc}",
        ) from exc

    # 7. Save history AFTER the response is sent. A slow or failing Supabase
    # can therefore never delay or destroy a successful analysis.
    background_tasks.add_task(_save_history_safely, user_id, filename, result)

    logger.info(
        "Request finished: %s | total_duration=%.2fs",
        filename,
        time.perf_counter() - request_start,
    )

    return response


@router.get("/history", tags=["History"])
async def get_history(
    user_id: str = Depends(get_current_user),
):
    """Return the signed-in user's analysis history."""

    try:
        from backend.database.supabase_db import get_user_history

        return await get_user_history(user_id)

    except Exception as exc:
        logger.exception("History fetch failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not load history: {exc}",
        ) from exc


@router.delete("/history/{analysis_id}", tags=["History"])
async def delete_history_entry(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete an analysis belonging to the signed-in user."""

    try:
        from backend.database.supabase_db import delete_analysis

        success = await delete_analysis(analysis_id, user_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Analysis not found or not owned by this user.",
            )

        return {"status": "deleted", "id": analysis_id}

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("History deletion failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete history: {exc}",
        ) from exc


@router.post("/generate-pdf", tags=["Reports"])
async def generate_pdf(
    data: AnalysisResponse,
    user_id: str = Depends(get_current_user),
):
    """Generate a PDF report from analysis data."""

    try:
        from fastapi.responses import Response
        from backend.services.report_generator import generate_html_reports
        from backend.services.pdf_export import generate_combined_pdf

        html_docs = await run_in_threadpool(
            generate_html_reports,
            data.model_dump(),
        )

        pdf_bytes = await run_in_threadpool(
            generate_combined_pdf,
            html_docs,
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="ats_report.pdf"'
            },
        )

    except Exception as exc:
        logger.exception("PDF generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF: {exc}",
        ) from exc


@router.get("/history/{analysis_id}/pdf", tags=["Reports"])
async def generate_history_pdf(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    """Generate a PDF for an existing analysis owned by the user."""

    try:
        from fastapi.responses import Response
        from backend.database.supabase_db import get_user_history
        from backend.services.report_generator import generate_html_reports
        from backend.services.pdf_export import generate_combined_pdf

        history = await get_user_history(user_id)

        analysis_data = next(
            (
                item.get("analysis_result")
                for item in history
                if str(item.get("id")) == str(analysis_id)
            ),
            None,
        )

        if not analysis_data:
            raise HTTPException(
                status_code=404,
                detail="Analysis not found.",
            )

        html_docs = await run_in_threadpool(
            generate_html_reports,
            analysis_data,
        )

        pdf_bytes = await run_in_threadpool(
            generate_combined_pdf,
            html_docs,
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ats_report_{analysis_id}.pdf"'
                )
            },
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("History PDF generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate history PDF: {exc}",
        ) from exc

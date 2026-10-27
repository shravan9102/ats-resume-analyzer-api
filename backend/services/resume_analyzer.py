import logging
import time
from typing import Dict, List, Optional

import spacy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any

from backend.models.schemas import IssueDetail
from backend.services.groq_parser import parse_resume, parse_job_description
from backend.services.jd_matcher import compare_resume_with_jd
from backend.services.feedback_engine import (
    analyze_issues,
    generate_issues_summary,
)
from backend.services.ats_scorer import (
    calculate_overall_score,
    validate_skills_with_projects,
)

logger = logging.getLogger("ats_resume_scorer")


def analyze_full_resume(
    resume_text: str,
    nlp: spacy.Language,
    embedder: SentenceTransformer,
    job_description: Optional[str] = None,
) -> Dict:
    """
    Analyze a resume and return ATS scores, feedback, and skill validation.
    Logs the duration of each major processing stage.
    """

    overall_start = time.perf_counter()

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Please upload a valid resume.")

    # Limit excessive input to reduce processing time.
    resume_text = resume_text[:20000]

    logger.info(
        "Resume analysis started. Extracted text length: %d characters",
        len(resume_text),
    )

    # 1. Parse resume using Groq.
    stage_start = time.perf_counter()
    logger.info("STEP 1: Starting resume parsing.")

    parsed_resume = parse_resume(resume_text)

    logger.info(
        "STEP 1 COMPLETE: Resume parsing took %.2f seconds.",
        time.perf_counter() - stage_start,
    )

    skills = parsed_resume.get("skills", [])
    projects = parsed_resume.get("projects", [])
    keywords = parsed_resume.get("keywords", [])
    action_verbs = parsed_resume.get("action_verbs", [])

    if not isinstance(skills, list):
        skills = []

    if not isinstance(projects, list):
        projects = []

    if not isinstance(keywords, list):
        keywords = []

    if not isinstance(action_verbs, list):
        action_verbs = []

    logger.info("Parsed skills count: %d", len(skills))
    logger.info("Parsed projects count: %d", len(projects))
    logger.info(
        "Parsed experience count: %d",
        len(parsed_resume.get("experience", [])),
    )

    # 2. Calculate total experience duration.
    experience_months = 0

    for experience in parsed_resume.get("experience", []):
        if not isinstance(experience, dict):
            continue

        try:
            duration = int(experience.get("duration_months", 0) or 0)
            experience_months += max(duration, 0)
        except (TypeError, ValueError):
            logger.warning("Invalid experience duration encountered.")

    # 3. Extract contact information.
    contact_info = {
        "email": parsed_resume.get("email"),
        "phone": parsed_resume.get("phone"),
        "linkedin": parsed_resume.get("linkedin"),
        "github": parsed_resume.get("github"),
        "portfolio": parsed_resume.get("portfolio"),
    }

    # 4. Validate skills against projects and experience.
    stage_start = time.perf_counter()
    logger.info("STEP 2: Starting skill validation.")

    skill_validation = validate_skills_with_projects(
        skills=skills,
        projects=projects,
        experience_entries=parsed_resume.get("experience", []),
        embedder=embedder,
    )

    logger.info(
        "STEP 2 COMPLETE: Skill validation took %.2f seconds.",
        time.perf_counter() - stage_start,
    )

    if not isinstance(skill_validation, dict):
        skill_validation = {}

    skill_validation.setdefault("validated_skills", [])
    skill_validation.setdefault("unvalidated_skills", [])

    # 5. Analyze job description if provided.
    jd_comparison_result = None
    jd_keywords = None

    if job_description and job_description.strip():
        clean_job_description = job_description.strip()[:12000]

        stage_start = time.perf_counter()
        logger.info("STEP 3: Starting job description parsing.")

        parsed_jd = parse_job_description(clean_job_description)

        logger.info(
            "STEP 3 COMPLETE: Job description parsing took %.2f seconds.",
            time.perf_counter() - stage_start,
        )

        required_skills = parsed_jd.get("required_skills", [])
        preferred_skills = parsed_jd.get("preferred_skills", [])
        extracted_keywords = parsed_jd.get("keywords", [])

        if not isinstance(required_skills, list):
            required_skills = []

        if not isinstance(preferred_skills, list):
            preferred_skills = []

        if not isinstance(extracted_keywords, list):
            extracted_keywords = []

        jd_keywords = list(
            dict.fromkeys(
                extracted_keywords + required_skills + preferred_skills
            )
        )

        stage_start = time.perf_counter()
        logger.info("STEP 4: Starting resume and JD comparison.")

        jd_comparison_result = compare_resume_with_jd(
            resume_text=resume_text,
            resume_keywords=keywords,
            resume_skills=skills,
            jd_text=clean_job_description,
            jd_keywords=jd_keywords,
            embedder=embedder,
            nlp=nlp,
        )

        logger.info(
            "STEP 4 COMPLETE: JD comparison took %.2f seconds.",
            time.perf_counter() - stage_start,
        )

    # 6. Load default grammar and location results.
    from backend.utils.file_utils import (
        get_default_grammar_results,
        get_default_location_results,
    )

    grammar_results = get_default_grammar_results()
    location_results = get_default_location_results()

    # 7. Calculate ATS scores.
    stage_start = time.perf_counter()
    logger.info("STEP 5: Calculating ATS scores.")

    scores = calculate_overall_score(
        text=resume_text,
        parsed_resume=parsed_resume,
        skills=skills,
        keywords=keywords,
        action_verbs=action_verbs,
        skill_validation_results=skill_validation,
        grammar_results=grammar_results,
        location_results=location_results,
        jd_keywords=jd_keywords,
        experience_months=experience_months,
    )

    logger.info(
        "STEP 5 COMPLETE: Score calculation took %.2f seconds.",
        time.perf_counter() - stage_start,
    )

    # 8. Generate detailed feedback.
    stage_start = time.perf_counter()
    logger.info("STEP 6: Generating detailed feedback.")

    detailed_feedback = analyze_issues(
        resume_text=resume_text,
        parsed_resume=parsed_resume,
        skills=skills,
        projects=projects,
        action_verbs=action_verbs,
        skill_validation=skill_validation,
        scores=scores,
        contact_info=contact_info,
    )

    issues_summary = generate_issues_summary(detailed_feedback)

    logger.info(
        "STEP 6 COMPLETE: Feedback generation took %.2f seconds.",
        time.perf_counter() - stage_start,
    )

    # 9. Prepare skill validation details.
    validated_raw = skill_validation.get("validated_skills", [])
    unvalidated_raw = skill_validation.get("unvalidated_skills", [])

    if not isinstance(validated_raw, list):
        validated_raw = []

    if not isinstance(unvalidated_raw, list):
        unvalidated_raw = []

    total_skills = len(validated_raw) + len(unvalidated_raw)

    validation_pct = round(
        len(validated_raw) / total_skills * 100,
        1,
    ) if total_skills else 0.0

    skill_validation_details = {
        "validated": [
            {
                "skill": item.get("skill", ""),
                "projects": item.get("projects", []),
            }
            for item in validated_raw
            if isinstance(item, dict)
        ],
        "unvalidated": unvalidated_raw,
        "total": total_skills,
        "validated_count": len(validated_raw),
        "validation_pct": validation_pct,
    }

    # 10. Build the final response.
    strengths = _generate_strengths(
        parsed_resume,
        skills,
        projects,
        action_verbs,
        skill_validation,
        scores,
    )

    matched_keywords = (
        jd_comparison_result.get("matched_keywords", [])
        if jd_comparison_result
        else keywords[:20]
    )

    missing_keywords = (
        jd_comparison_result.get("missing_keywords", [])
        if jd_comparison_result
        else []
    )

    result = {
        "ATS_score": scores["overall_score"],
        "ats_score": scores["overall_score"],
        "component_scores": {
            "formatting": scores["formatting_score"],
            "keywords": scores["keywords_score"],
            "content": scores["content_score"],
            "skill_validation": scores["skill_validation_score"],
            "ats_compatibility": scores["ats_compatibility_score"],
        },
        "issues_summary": issues_summary,
        "detailed_feedback": detailed_feedback,
        "jd_match_analysis": jd_comparison_result,
        "jd_comparison": jd_comparison_result,
        "skills": skills,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "strengths": strengths,
        "interpretation": scores.get("overall_interpretation", ""),
        "skill_validation_details": skill_validation_details,
        "experience_months": experience_months,
    }

    logger.info(
        "Resume analysis completed successfully in %.2f seconds.",
        time.perf_counter() - overall_start,
    )

    return result


def _generate_strengths(
    parsed_resume: Dict,
    skills: List,
    projects: List,
    action_verbs: List,
    skill_validation: Dict,
    scores: Dict,
) -> List[str]:
    """Generate strengths based on structured resume data."""

    strengths = []

    if parsed_resume.get("experience"):
        strengths.append("Has a dedicated Experience section")

    if parsed_resume.get("projects") or projects:
        strengths.append(
            "Includes a Projects section showcasing applied skills"
        )

    if parsed_resume.get("education"):
        strengths.append("Education section is present")

    if skills:
        strengths.append("Clear Skills section with listed technologies")

    summary = parsed_resume.get("professional_summary", "")
    if isinstance(summary, str) and summary.strip():
        strengths.append(
            "Professional Summary provides a quick overview"
        )

    if len(skills) >= 8:
        strengths.append(
            f"Strong skill set — {len(skills)} skills detected"
        )

    if len(action_verbs) >= 5:
        strengths.append(
            f"Uses {len(action_verbs)} strong action verbs in bullet points"
        )

    validated = skill_validation.get("validated_skills", [])

    if isinstance(validated, list) and len(validated) >= 3:
        strengths.append(
            f"{len(validated)} skills are backed by project/experience evidence"
        )

    if scores.get("formatting_score", 0) >= 16:
        strengths.append("Well-formatted and ATS-friendly structure")

    if scores.get("content_score", 0) >= 20:
        strengths.append(
            "Content quality is high with measurable achievements"
        )

    return strengths

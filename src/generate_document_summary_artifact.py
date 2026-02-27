from assemble_output import document_summary_artifact_lambda_handler


def lambda_handler(event: dict, context: object) -> dict:
    return document_summary_artifact_lambda_handler(event, context)

from assemble_output import authorship_artifact_lambda_handler


def lambda_handler(event: dict, context: object) -> dict:
    return authorship_artifact_lambda_handler(event, context)

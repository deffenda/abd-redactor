from assemble_output import assemble_redaction_lambda_handler


def lambda_handler(event: dict, context: object) -> dict:
    return assemble_redaction_lambda_handler(event, context)

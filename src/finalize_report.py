from assemble_output import finalize_report_lambda_handler


def lambda_handler(event: dict, context: object) -> dict:
    return finalize_report_lambda_handler(event, context)

import sys

def error_message_detail(error, error_detail:sys):
  ## we are not interested first two paramenters hence _
  ## exc_tb ll tell on which file, which line the exception has occured
  _,_,exc_tb=error_detail.exc_info() 
  file_name = exc_tb.tb_frame.f_code.co_filename
  error_message = "Error at file: {0} || line: {1} || error: {2}".format(
    file_name, exc_tb.tb_lineno, str(error)
  )
  return error_message


class CustomException(Exception):
  def __init__(self, error_message, error_detail:sys):
    super().__init__(error_message)
    self.error_message=error_message_detail(error_message,error_detail=error_detail)

  def __str__(self):
    return self.error_message
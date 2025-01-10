from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirement(file_path: str) -> List[str]:
  '''
  import list of pacakges from filepaths
  '''

  requirements=[]
  with open(file_path) as file_obj:
    requirements= file_obj.readlines() 
    
    ## at the end of each lines \n is appended, which we need to remove
    requirements = [req.replace("\n","") for req in requirements]

    ## when running packages names from file, we need to avoid "-e ."
    if HYPHEN_E_DOT in requirements:
      requirements.remove(HYPHEN_E_DOT)
  
  return requirements

'''
On running pip install -r requirements.txt
setup.py ll also gets triggered because of "-e ." in requirements.txt
'''
setup(
  name='mlproject',
  version='0.0.1',
  author='sumeet',
  author_email='sumeet.030990@gmail.com',
  packages=find_packages(),
  install_requires=get_requirement("requirements.txt")
)
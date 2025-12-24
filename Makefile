execute :
	poetry run main-api

execute-api :
	fastapi dev 
	
execute-training:
	poetry run python src/mon_mlops_project/training.py
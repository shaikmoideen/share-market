<!-- 
News API Key - 612a7281b32f4a9abf0edc18aae5effd 
Alpha Vantage API Key - W2653K1FAP713260
-->

How To Set Daily Retraining in Windows

Go To: Task Scheduler

STEP 1
    Click - Create Basic Task

STEP 2
    Name - Daily Stock Model Training

STEP 3
    Choose - Daily

STEP 4
    Set Time - 6:00 AM

STEP 5
    Choose - Start a Program

STEP 6
    Program/script:
    python

    or full path like:
    C:\Python314\python.exe

STEP 7
    Add arguments:
    train_model.py

STEP 8
    Start in:
    C:\Users\YourFolder\stock_ai_project

DONE ✅
    Now every day:

    6 AM
    ↓
    model retrains automatically
    ↓
    dashboard uses fresh model
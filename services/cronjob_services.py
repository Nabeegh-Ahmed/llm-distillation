import schedule
def queue_job(job, time):
    schedule.every(time).minutes.do(job)

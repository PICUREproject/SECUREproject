import tkinter as tk
import subprocess
import webbrowser
import time

def start_server():
    subprocess.Popen(["python", "-m", "uvicorn", "Fastapi_run.server2:app", "--reload"], cwd="C:/Users/LEEHoyJoung/project/atm")
    
    
    time.sleep(2)  
    webbrowser.open("http://127.0.0.1:8000")

root = tk.Tk()
root.title("FastAPI Server")
root.geometry("800x600")  


start_button = tk.Button(
    root, 
    text="PHICURE ATM 시작", 
    command=start_server, 
    width=18,                      
    height=3,                      
    font=("Helvetica", 30, "bold"), 
    bg="#CECEFF",                  
    fg="black",                    
    activebackground="#6D6DC6",    
    activeforeground="black",      
    relief="solid",                
    bd=5,                          
)

start_button.place(relx=0.5, rely=0.5, anchor="center")


root.mainloop()
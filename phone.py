import webbrowser
import http.server
import socketserver


PORT = 8000

html_content = '''
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>경고 메시지</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body, html {
      height: 100%;
      display: flex;
      flex-direction: column;
      font-family: Arial, sans-serif;
      background-color: #f5f5f5;
      justify-content: center;
      align-items: center;
    }

    .header {
      background-color: #ffd966;
      width: 100%;
      text-align: center;
      padding: 30px 0;
      font-size: 3.5rem;
      font-weight: bold;
      color: #333;
      border-bottom: 3px solid #ccc;
    }

    .container {
      width: 100%;
      max-width: 1380px;
      padding: 60px;
      text-align: center;
      background-color: #e0e7ef;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 40px;
    }

    .message-box {
      width: 100%;
      padding: 50px;
      background-color: white;
      border-radius: 10px;
      font-size: 3.5rem;
      font-weight: bold;
      color: #0d0d0df1;
      line-height: 1.5;
      border: 2px solid #ccc;
      animation: blink 1.5s infinite;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .message-box p {
      margin-bottom: 20px; 
    }

    .button {
      padding: 20px 60px;
      font-size: 2.5rem;
      font-weight: bold;
      cursor: pointer;
      border: none;
      border-radius: 10px;
      color: white;
      background-color: #4CAF50;
      transition: background-color 0.3s ease;
    }

    .button:hover {
      background-color: #45a049;
    }

    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
  </style>

  <script>

    function playVoice() {
      const message = ".....거래를 위해 본인인증을 진행해주십시오.";
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.lang = "ko-KR";
      utterance.rate = 1.4;
      window.speechSynthesis.speak(utterance);
    }

    window.addEventListener("load", function () {
      playVoice(); 
      setTimeout(function () {
        window.close(); 
      }, 5000);
    });
    
  </script>
  </head>
  <body>
  
    <div class="header">안 내</div>
  
    <div class="container">
      <div class="message-box">
        <p>거래에 문제가 발생했습니다.</p>
        <p>본인인증을 진행해 주세요.</p>
      </div>
  
      <button class="button" onclick="window.close()">계속 거래합니다</button>
    </div>
  </body>
</html>


'''

def start_phone_server():
  with open('social_login.html', 'w', encoding='utf-8') as file:
    file.write(html_content)

  # 기본 웹 브라우저에서 HTML 파일을 엽니다.
  class Handler(http.server.SimpleHTTPRequestHandler):
        
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

  # 현재 디렉토리에서 HTTP 서버를 시작합니다.
  with socketserver.TCPServer(("", PORT), Handler) as httpd:
      print("serving at port", PORT)
      webbrowser.open(f"http://localhost:{PORT}/social_login.html")
      httpd.serve_forever()
  webbrowser.open('social_login.html')
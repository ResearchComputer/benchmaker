from flash_sandbox import HTTPClient
client = HTTPClient(address="https://sandbox.swissai.cscs.ch")
sb = client.start_sandbox(type="kubernetes", image="alpine:3.20",
                        command=["sh","-c","sleep 60"])
print(sb.exec_command(["ls", "/"]).stdout)
sb.stop()
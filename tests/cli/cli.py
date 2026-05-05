"""Simple CLI client for testing Robot Bridge"""
import asyncio
import os
import argparse


class RobotBridgeCLI:
    """Simple CLI for interacting with Robot Bridge"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8081"):
        self.base_url = base_url
        self.session_id = None
    
    async def chat(self, message: str):
        """Send chat message"""
        import httpx
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "message": message,
                    "session_id": self.session_id,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                return data.get("text", "")
            else:
                raise Exception(f"Error: {response.status_code} - {response.text}")
    
    async def tts(self, text: str, output_file: str = None):
        """Synthesize speech"""
        import httpx
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/api/tts",
                json={"text": text}
            )
            
            if response.status_code == 200:
                if output_file:
                    with open(output_file, "wb") as f:
                        f.write(response.content)
                    return output_file
                else:
                    return response.content
            else:
                raise Exception(f"Error: {response.status_code}")
    
    async def tts_and_play(self, text: str):
        """Synthesize and play audio"""
        import tempfile
        import subprocess
        
        # Generate TTS
        audio = await self.tts(text)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio)
            temp_path = f.name
        
        try:
            # Play with ffplay or mpv
            players = ["ffplay", "mpv", "mpg123"]
            for player in players:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        player, "-nodisp", "-autoexit", temp_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await proc.communicate()
                    break
                except FileNotFoundError:
                    continue
            else:
                print(f"TTS generated but no player found. Saved to {temp_path}")
        finally:
            os.unlink(temp_path)
    
    async def interactive(self):
        """Interactive chat mode"""
        print("\n" + "=" * 50)
        print("Robot Bridge Interactive Mode")
        print("Type 'quit' or 'exit' to exit")
        print("Type 'tts <text>' to synthesize speech")
        print("=" * 50 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                
                if user_input.lower().startswith("tts "):
                    text = user_input[4:]
                    print("Synthesizing...")
                    await self.tts_and_play(text)
                    continue
                
                if user_input.lower() == "tts":
                    print("Usage: tts <text>")
                    continue
                
                # Regular chat
                print("Thinking...")
                response = await self.chat(user_input)
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


async def main():
    parser = argparse.ArgumentParser(description="Robot Bridge CLI")
    parser.add_argument("--url", default="http://127.0.0.1:8081", 
                        help="Robot Bridge URL")
    parser.add_argument("--chat", help="Send single chat message")
    parser.add_argument("--tts", help="Synthesize speech")
    parser.add_argument("--play", action="store_true", help="Play TTS audio")
    
    args = parser.parse_args()
    
    cli = RobotBridgeCLI(args.url)
    
    if args.chat:
        response = await cli.chat(args.chat)
        print(response)
    elif args.tts:
        if args.play:
            await cli.tts_and_play(args.tts)
        else:
            audio = await cli.tts(args.tts)
            print(f"Generated {len(audio)} bytes of audio")
    else:
        await cli.interactive()


if __name__ == "__main__":
    asyncio.run(main())

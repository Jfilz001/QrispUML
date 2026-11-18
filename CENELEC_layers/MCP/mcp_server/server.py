from git import Repo
import mcp.server
import asyncio
import json
import sys
import ast

class QrispClassDiffer:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        
    def get_class_changes(self, old_commit, new_commit):
        try:
            repo = Repo(self.repo_path)
            classes_removed = set()
            classes_added = set()
            classes_modified = []
            
            for patch in repo.diff(old_commit, new_commit, create_patch=True):
                if patch.a_path.endswith('.py') and not patch.a_path.startswith('documentation/'):
                    old_content = patch.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                    new_content = patch.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                    
                    old_class_names = {node.name for node in ast.walk(ast.parse(old_content)) if isinstance(node, ast.ClassDef)}
                    new_class_names = {node.name for node in ast.walk(ast.parse(new_content)) if isinstance(node, ast.ClassDef)}
                    
                    classes_removed.update([c for c in old_class_names - new_class_names])
                    classes_added.update([c for c in new_class_names - old_class_names])
                    classes_modified.extend([{'file': patch.a_path, 'class': c} 
                    for c in old_class_names & new_class_names if old_content != new_content])
                    
            return {
                "status": "ok",
                "classes_removed": list(classes_removed)[:100],
                "classes_added": list(classes_added)[:100],
                "classes_modified": classes_modified[:50]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    def get_all_classes(self):
        try:
            repo = Repo(self.repo_path)
            classes = []
            
            for item in repo.tree().traverse():
                if isinstance(item, (str,)) and item.endswith('.py'):
                    if not item.startswith('documentation/'):
                        content = repo.git.show("{repo.head.commit}:{item}")
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                classes.append(f"{item}.{node.name}")
            
            return {"status": "ok", "classes": sorted(list(set(classes)))[:200]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_class_methods(self, class_name, commit):
        try:
            repo = Repo(self.repo_path)
            
            for item in repo.tree().traverse():
                if isinstance(item, (str,)) and item.endswith('.py'):
                    if not item.startswith('documentation/'):
                        content = repo.git.show(f"{commit}:{item}")
                        tree_ast = ast.parse(content)
                        
                        for node in ast.walk(tree_ast):
                            if isinstance(node, ast.ClassDef) and node.name == class_name:
                                methods = []
                                for n in ast.walk(node):
                                    if isinstance(n, (ast.FunctionDef,)):
                                        args = [arg.arg for arg in n.args.args]
                                        methods.append({"name": n.name, "args": args})
                                
                                return {"status": "ok", "class": class_name, "methods": methods}
            
            return {"status": "not_found", "message": f"Class {class_name} not found in commit {commit[:7]}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

import socket

async def main():
    repo_path = r"C:\Users\filzj\Desktop\Hiwi\QrispUML"
    
    # Check if already running by trying to bind to port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', 8765))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.listen(1)
        sock.close()
    except OSError:
        print("Server already running on port 8765")
        return
    
    tools_config = [
        {"name": "get_class_changes", "description": "Get classes removed, added, and modified between two git commits."},
        {"name": "get_all_classes", "description": "Get all class names in the current repository version."},
        {"name": "get_class_methods", "description": "Get methods for a specific class at a given commit."}
    ]

    await asyncio.get_event_loop().run_in_executor(None, print, json.dumps({
        "jsonrpc": "2.0", "id": 1, "result": {
            "protocolVersion": "2024-11-05", 
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "qrisp-class-differ", "version": "1.0.0"}
        }
    }))

    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            message = json.loads(line.strip())
            
            if message["method"] == "tools/list":
                await asyncio.get_event_loop().run_in_executor(None, print, json.dumps({
                    "jsonrpc": "2.0", 
                    "id": message["id"], 
                    "result": {"tools": tools_config}
                }))
            
            elif message["method"] == "tools/call" and message["params"]:
                params = message["params"]["arguments"]
                differ = QrispClassDiffer(repo_path)
                
                if message["params"]["name"] == "get_class_changes":
                    result = differ.get_class_changes(params[0], params[1])
                elif message["params"]["name"] == "get_all_classes":
                    result = differ.get_all_classes()
                elif message["params"]["name"] == "get_class_methods":
                    result = differ.get_class_methods(params[0], params[1])
                
                await asyncio.get_event_loop().run_in_executor(None, print, json.dumps({
                    "jsonrpc": "2.0", 
                    "id": message["id"], 
                    "result": {"content": [{"type": "text", "text": json.dumps(result)}]}
                }))
        except Exception as e:
            await asyncio.get_event_loop().run_in_executor(None, print, str(e))

if __name__ == "__main__":
    asyncio.run(main())

import ast
import json
import os
from typing import List, Dict, Any, Optional
import git

class ClassAttribute:
    def __init__(self, name: str, data_type: str = "Any", visibility: str = "public", static: bool = False):
        self.name = name
        self.data_type = data_type
        self.visibility = visibility
        self.static = static

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.data_type,
            "visibility": self.visibility,
            "static": self.static
        }

class ClassMethod:
    def __init__(self, name: str, return_type: str = "None", parameters: List[Dict[str, str]] = None, visibility: str = "public", static: bool = False):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters or []
        self.visibility = visibility
        self.static = static

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "visibility": self.visibility,
            "static": self.static
        }

class ClassDefinition:
    def __init__(self, class_name: str, parents: List[str], attributes: List[ClassAttribute], methods: List[ClassMethod], abstract: bool = False):
        self.class_name = class_name
        self.parents = parents
        self.attributes = attributes
        self.methods = methods
        self.abstract = abstract

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "parents": self.parents,
            "abstract": self.abstract,
            "attributes": [attr.to_dict() for attr in self.attributes],
            "methods": [method.to_dict() for method in self.methods]
        }

def determine_visibility(name: str) -> str:
    """Bestimmt die Sichtbarkeit basierend auf Python-Namenskonventionen."""
    if name.startswith("__") and not name.endswith("__"):
        return "private"
    elif name.startswith("_"):
        return "protected"
    return "public"

def is_static_method(decorator_list: List[ast.expr]) -> bool:
    """Prüft, ob eine Methode mit @staticmethod oder @classmethod dekoriert ist."""
    for dec in decorator_list:
        if isinstance(dec, ast.Name) and dec.id in ("classmethod", "staticmethod"):
            return True
        elif isinstance(dec, ast.Attribute) and dec.attr in ("classmethod", "staticmethod"):
            return True
    return False

def parse_python_file(file_path: str) -> List[ClassDefinition]:
    """Parst eine .py Datei und gibt eine Liste von ClassDefinition Objekten zurück."""
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            tree = ast.parse(file.read(), filename=file_path)
        except SyntaxError:
            return []

    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 1. Elternklassen (Base Classes) extrahieren & prüfen ob Abstract
            parents = []
            is_abstract_class = False
            for base in node.bases:
                if isinstance(base, ast.Name):
                    parents.append(base.id)
                    if base.id == "ABC":
                        is_abstract_class = True
                elif isinstance(base, ast.Attribute):
                    parents.append(f"{ast.unparse(base.value)}.{base.attr}")
                    if base.attr == "ABC":
                        is_abstract_class = True

            attributes_dict: Dict[str, ClassAttribute] = {}
            methods: List[ClassMethod] = []

            for item in node.body:
                # 2. Klassenattribute (z.B. x = 10) -> Statisch!
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            attr_name = target.id
                            attributes_dict[attr_name] = ClassAttribute(
                                name=attr_name,
                                data_type="Any",
                                visibility=determine_visibility(attr_name),
                                static=True
                            )

                # 3. Typ-annotierte Klassenattribute (z.B. x: int = 10) -> Statisch!
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        attr_name = item.target.id
                        type_str = ast.unparse(item.annotation) if item.annotation else "Any"
                        attributes_dict[attr_name] = ClassAttribute(
                            name=attr_name,
                            data_type=type_str,
                            visibility=determine_visibility(attr_name),
                            static=True
                        )

                # 4. Methoden & Instanzattribute (aus __init__ / Methoden)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = item.name
                    return_type = ast.unparse(item.returns) if item.returns else "None"
                    
                    # Prüfen ob @classmethod oder @staticmethod vorhanden ist
                    method_is_static = is_static_method(item.decorator_list)

                    # Parameter extrahieren
                    params = []
                    for arg in item.args.args:
                        param_name = arg.arg
                        param_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
                        params.append({"name": param_name, "type": param_type})

                    methods.append(ClassMethod(
                        name=method_name,
                        return_type=return_type,
                        parameters=params,
                        visibility=determine_visibility(method_name),
                        static=method_is_static
                    ))

                    # Instanzattribute aus `self.attr = ...` -> Nicht statisch (static=False)
                    for sub_node in ast.walk(item):
                        if isinstance(sub_node, ast.Assign):
                            for target in sub_node.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                    attr_name = target.attr
                                    if attr_name not in attributes_dict:
                                        attributes_dict[attr_name] = ClassAttribute(
                                            name=attr_name,
                                            data_type="Any",
                                            visibility=determine_visibility(attr_name),
                                            static=False
                                        )

                        elif isinstance(sub_node, ast.AnnAssign):
                            if isinstance(sub_node.target, ast.Attribute) and isinstance(sub_node.target.value, ast.Name) and sub_node.target.value.id == "self":
                                attr_name = sub_node.target.attr
                                type_str = ast.unparse(sub_node.annotation) if sub_node.annotation else "Any"
                                attributes_dict[attr_name] = ClassAttribute(
                                    name=attr_name,
                                    data_type=type_str,
                                    visibility=determine_visibility(attr_name),
                                    static=False
                                )

            classes.append(ClassDefinition(
                class_name=node.name,
                parents=parents,
                attributes=list(attributes_dict.values()),
                methods=methods,
                abstract=is_abstract_class
            ))

    return classes

def scan_repository(repo_path: str) -> List[Dict[str, Any]]:
    """Scannt ein ganzes Git-Repository nach .py-Dateien und extrahiert alle Klassen."""
    all_classes = []
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                classes = parse_python_file(full_path)
                for cls in classes:
                    cls_dict = cls.to_dict()
                    cls_dict["file_path"] = os.path.relpath(full_path, repo_path)
                    all_classes.append(cls_dict)
                    
    return all_classes

# ==========================================
# ANWENDUNG
# ==========================================
if __name__ == "__main__":
    REPO_PATH = r"C:\Users\filzj\Hiwi\Qrisp"
    OUTPUT_JSON_PATH = "qrisp_uml_data.json"

    version = git.Repo(REPO_PATH, search_parent_directories=True).head.commit.hexsha
    
    # Repository scannen und Klassen extrahieren
    extracted_data = scan_repository(os.path.join(REPO_PATH, "src"))
    
    output_payload = {
        "commit_hash": version,
        "classes": extracted_data
    }

    # Als JSON speichern
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as out_file:
        json.dump(output_payload, out_file, indent=4, ensure_ascii=False)

    print(f"Erfolgreich {len(extracted_data)} Klassen nach '{OUTPUT_JSON_PATH}' exportiert.")
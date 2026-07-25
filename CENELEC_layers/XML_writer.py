import xml.etree.ElementTree as ET


class XML_Handler():
    
    def __init__(self, *args, **kwargs):
        self.xml_data = self.load_xml()
        self.titles_to_ignore = ["Programming Layer:", "Assembly layer:", "HAL:"]

    
    def load_data(self):
        try:
            with open(r"C:\Users\filzj\Desktop\Hiwi\CENELEC_LAYERS_QRISP.drawio", "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            print("Falsche Datei")
        except PermissionError:
            print("Keine Berechtigung")  
        
    def read_classes(self):
        classes={}    
        try:
            with open(r"C:\Users\filzj\Desktop\Hiwi\Classes.txt", "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file.readlines()]
                for line in lines:   
                    if line in self.titles_to_ignore:
                        index=line
                        classes[index] = []
                    else:
                        line = line.strip("-\t")
                        classes[index].append(line)
                
        except FileNotFoundError:
            print("Falsche Datei")
        except PermissionError:
            print("Keine Berechtigung")
        return classes
        

    def load_xml(self):
        xml_data = self.load_data()
        root = ET.fromstring(xml_data)
        graph_root = root[0][0][0]   
        return graph_root
        
    def get_latest_elements(self,graph_root):
        for element in graph_root:
            attributes = element.attrib
            if(attributes["id"].startswith("NwllD8PpyCOOfQ-9G6Aj") and attributes["value"] not in self.titles_to_ignore):
                latest_name = attributes
            for x in element:
                latest_x = x.attrib
        return latest_name, latest_x

    def get_x(self):
        _,x = self.get_latest_elements(self.xml_data)
        return int(x["x"])
        
    def calc_element_width(self,text):
        laenge = len(text)
        if(laenge<=10):
            width=80
        elif(laenge>10 and laenge<=13):
            width=100
        elif(laenge>13 and laenge<=17):
            width=120
        elif(laenge>17 and laenge<=20):
            width=140
        elif(laenge>20 and laenge<=23):
            width=160
        elif(laenge>23):
            width = 200
        return width        
        
    
    def get_element_template(self):
        id, _ = self.get_latest_elements(self.xml_data)
        index = int(id["id"].strip("NwllD8PpyCOOfQ-9G6Aj-"))+1
        classes = self.read_classes()
        y=240
        spacing = 40
        strings_to_add = {}
        for i in classes:
            x = self.get_x()
            strings_to_add[i] = []
            pre_width = 0
            for j in classes[i]:
                width = self.calc_element_width(j)
                x = x + pre_width + spacing
                temp_str = (f"<mxCell id=\"NwllD8PpyCOOfQ-9G6Aj-{index}\" parent=\"1\" style=\"html=1;whiteSpace=wrap;\" value=\"{j}\" vertex=\"1\"> \n" \
                        f"<mxGeometry height=\"40\" width=\"{width}\" x=\"{x}\" y=\"{y}\" as=\"geometry\" /> \n" \
                    f"</mxCell>")
                strings_to_add[i].append(temp_str)
                index+=1
                pre_width = width
            y+=120
        return strings_to_add
        
            

    def write_elements(self):
        self.get_latest_elements(self.xml_data)
        
        for i, e in enumerate(self.xml_data):
            if(i>2):
                print(i, self.get_x_y(e))
                
                
    
handler = XML_Handler()

values = handler.get_element_template()

for group in values:
    for e in values[group]:
        print(e)





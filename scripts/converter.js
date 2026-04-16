const fs = require("fs");
const { convertDocxToHtml } = require("docx-parser-converter");

function extractSection(html, name) {
    const regex = new RegExp(
        `<!-- START: ${name} -->([\\s\\S]*?)<!-- END: ${name} -->`,
        "i"
    );
    const match = html.match(regex);
    return match ? match[1].trim() : "";
}

(async () => {
    const buffer = fs.readFileSync("./documentation/classes/Documentation.docx");
    const html = await convertDocxToHtml(buffer);



    //const sections = ["Backends", "Hamiltonian", "Metrics", "Operations","Permeability","QEnv_QPrim","QuantumManagement","References","Tensor"];

    const sections = ["test"]

    sections.forEach(section => {

        const content = extractSection(html, section);

        const output = template.replace(
            "<!-- DOCX_CONTENT -->",
            content
        );

        fs.writeFileSync(`./documentation/classes/${section}.html`, output);
    });
})();
 <%@ LANGUAGE="VBScript" %>
<html>
<head>
    <title>TAAA Model</title>
    <meta charset="utf-8" />
</head>
<body>
    <h2>🔢 TAAA Rasch Model Page</h2>
<%
    Dim score, diff, theta
    score = 0.8
    diff = 1.2
    theta = Log(score / (1 - score)) - diff
    Response.Write("<p>θ (ability estimate): " & Round(theta, 3) & "</p>")
%>
</body>
</html>

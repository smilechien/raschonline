 
   <!-- #include file="../ADOFunctions.asp" --> 
    <% 'Response.Buffer = False %>
    <%Response.Buffer=false%>
 
 <html xmlns="http://www.w3.org/1999/xhtml">
 <head>
<meta http-equiv="Content-Type" content="text/html; charset=big5" />
 
<title>RaschOnline</title>
<meta name="viewport" content="width=device-width; initial-scale=1.0">
 
<scgript src="../autoadjust/js/jquery-1.7.1.min.js"></script>
<script src="../autoadjust/js/script.js"></script>
<script src="../autoadjust/js/forms.js"></script>
<script src="../autoadjust/js/superfish.js"></script>
<script src="../autoadjust/js/jquery.responsivemenu.js"></script>
<script src="../autoadjust/js/FF-cash.js"></script>
<script>
$(function(){
	$('#contact-form').forms({ownerEmail:'#'});
});
</script>
 
<!DOCTYPE html>
<!--[if lt IE 7 ]> <html lang="zh-tw" class="no-js ie6"> <![endif]-->
<!--[if IE 7 ]>    <html lang="zh-tw" class="no-js ie7"> <![endif]-->
<!--[if IE 8 ]>    <html lang="zh-tw" class="no-js ie8"> <![endif]-->
<!--[if IE 9 ]>    <html lang="zh-tw" class="no-js ie9"> <![endif]-->
<!--[if (gt IE 9)|!(IE)]><!-->
<html lang="zh-tw" class="no-js"> <!--<![endif]-->
<html lang="en">
<head>
    <meta charset="UTF-8">
	<title>PaperABC</title>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8" /> 
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<link rel="stylesheet" href="css/style.css" type="text/css" />
    <!--[if IE]>
	<script src="js/html5shiv.js"></script>
	<![endif]-->
	<script src="js/jquery-1.7.1.min.js"></script>
	<script src="js/modernizr.custom.js"></script>
	<link rel="stylesheet" type="text/css" href="css/demo.css" />
	<link rel="stylesheet" type="text/css" href="css/component.css" />
	<script src="js/jquery.nav.js"></script>
<!--link rel="stylesheet" href="../css/stylesheet-image-based.css"--> 
<style>
 .radio {
  position: relative;
  float: left;
  clear: left;
  display: block;
  padding-left: 40px;
  margin-bottom: 12px;
  line-height: 22px;
  font-size: 22px;
  color: #666;
  cursor: pointer;
}
.radio:before {
  background: #fff;
  content: "";
  position: absolute;
  display: inline-block;
  top: 0;
  left: 0;
  width: 22px;
  height: 21px;
  border: 1px solid #bbb;

  border-radius: 100%;
  -moz-border-radius: 100%;
  -webkit-border-radius: 100%;

  box-shadow: inset 0 0 3px 0 #ccc;
  -moz-box-shadow: inset 0 0 3px 0 #ccc;
  -webkit-box-shadow: inset 0 0 3px 0 #ccc;
}
input[type="radio"] {
  display: none;
} 
input[type="radio"]:checked + label:before {
  content: "\2022";
  text-align: center;
  line-height: 15px;
  font-family: Tahoma; /* If you change font you must also change font-size & line-height */
  font-size: 44px;
  color: #00a0db;
  text-shadow: 0 0 4px #bbb;
}


 
.Checkbox {
  position: relative;
  float: left;
  clear: left;
  display: block;
  padding-left: 40px;
  margin-bottom: 12px;
  line-height: 22px;
  font-size: 22px;
  color: #666;
  cursor: pointer;
}

 
.Checkbox input {
    display: none;
}

.Checkbox span {
 
background: #fff;
  content: "";
  position: absolute;
  display: inline-block;
  top: 0;
  left: 0;
  width: 22px;
  height: 21px;
  border: 1px solid #bbb;

  background: url("../css/checks.png");

  box-shadow: inset 0 0 3px 0 #ccc;
  -moz-box-shadow: inset 0 0 3px 0 #ccc;
  -webkit-box-shadow: inset 0 0 3px 0 #ccc;
}

.Checkbox input:checked + span {
    background: url("../css/checks.png");
}

 
</style>
<script language="javascript" type="text/javascript"> 
function goAndClose(url) {
  opener.location.href = url
  this.close;
}
</script>
 <style>
   .red {
       color: red;
       font-size: 2em;
  }
  .blue {
       color: blue;
       font-weight: bold;
   }
  </style>
<script language="javascript" type="text/javascript"> 
 function clearContent()
{
    document.getElementById("output").value='';
}
</script>
	 <style>				
		 body {				
		   background-color:#E7E9EB;				
		 }				
		 #myDIV {				
		   height:300px;				
		   background-color:#FFFFFF;				
		 }				
		 #blueDIV {				
		   position:relative;				
		   width:100px;				
		   padding:20px; 				
		   transform:  rotate(15deg);				
		 }				
		 #chienDIV {				
		 .translate-rotate {				
		   transform: translateX(180px) rotate(45deg);				
		 }				
	 
h1 {
  font-size: 2.5em; /* 40px/16=2.5em */
}

h2 {
  font-size: 1.875em; /* 30px/16=1.875em */
 }

p {
  font-size: 0.875em; /* 14px/16=0.875em */
}
</style>
       <script>
            function Geeks() {
                var doc = document.getElementById('text2');
                doc.select();
            }
        </script> 
</head>
<body  style="background-color:white;"> <div align="center">
 <h3><a href="#bottom">Bottom of document</a></a></h3>

<%
personm=request("personm")
itemd=request("itemd") 
        
repno=request("repno")
covid=request("covid")
kid=request("kid")
if kid="" then kid=1

if repno="" then repno=1
myear=request("myear")
content= request("content")
'content=  trim(content) 
 for jk=len(content) to 1
   if right(content,1)=chr(32) then
      content=mid(content,1,len(content)-1)
   else
      exit for
    end if

  next 

 if len(content)>20 then
  if mid(content,len(content),1)=chr(13) then
   response.write instr(content, chr(13)) & ":" & content
   response.end
          content=trim(mid(content,len(content)-1,1))
  end if
 
 if mid(content,len(content),1)=chr(10) then
          content=trim(mid(content,1,len(content)-1))
  end if
 end if
covid=request("covid")
 'response.write content & "ddd"
 
  
  if  (content="" or content="Enter text here...") then
    %>
     <center>
 <form id="form1" method="post" action="raschrsm.asp"  name="post"  >
          
 <table><tr><td>  <font size=18><font color=blue>
  </font></td><td>           
 <font color=red>Copy & Paste data<br>dot for missing</font> 
<textarea id='output' rows="15" name="content" cols="60">
  <% if  Session("content")="" then %>

Item1	item2	Item3	Item4	Item5	Item6	Item7	Item8	Item9	Item10	name	group
.	1	1	1	1	1	1	1	0	1	Student1	1
1	1	1	1	1	1	1	0	1	0	Student2	1
1	1	1	1	1	1	0	1	0	0	Student3	1
1	1	1	1	1	1	0	1	0	0	Student4	0
1	1	1	1	1	1	0	1	0	0	Student5	0
1	1	1	1	1	0	1	0	0	0	Student6	1
1	1	1	1	0	1	0	0	0	0	Student7	0
1	0	1	0	1	0	0	0	0	0	Student8	1
0	1	0	1	0	0	0	0	0	0	Student9	0

 <% else%>
  
   <% response.write trim(Session("content"))
  end if %>
</textarea></td></tr></table>
  <input type="button" value="Clear" onclick="clearContent()" style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px">   

<input type="button" name="button" value="Back" onClick="location.href='index.html'" style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px">
<input type="button" name="button" value="Refresh" onClick="location.href='raschrsm.asp'" style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px">
<input type="button" name="button" value="Nil Session" onClick="location.href='nilsession.asp'" style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px"><br>   


<br> 
<font color=red> 
 <input type="hidden"  name="categoryabc" value="2" > 
 <% ' <input type="hidden"  name="logtrans" value="4"> %>    
<font color=red>Transform data</font> <SELECT NAME="logtrans"> 
     <OPTION VALUE="1" >Log
     <OPTION VALUE="2">Linear
     <OPTION VALUE="3">Percentage
     <OPTION VALUE="4" SELECTED >Category
</SELECT> <BR>
  <input type="hidden"  name="itemd" value="<%=request("itemd")%>"> 
<input type="hidden"  name="personm" value="<%=request("personm")%>"> 
<font color=red>Visual displays</font> <SELECT NAME="covid">

     <OPTION VALUE="">None
 <% if covid="33" and request("personm")="1" then %>
 <OPTION VALUE="33" SELECTED> CBP_person
 <% elseif covid="33" and request("personm")="2" then %>
<OPTION VALUE="33" SELECTED>CBP_POutfit
  <% elseif covid="33" and request("personm")="3" then %>
<OPTION VALUE="33" SELECTED>CBP_PInfit   
<% elseif covid="34" and request("itemd")="1" then%>
<OPTION VALUE="34" SELECTED>CBP_item
 <% elseif covid="34" and request("itemd")="2" then%>
<OPTION VALUE="34" SELECTED>CBP_iOutfit
<% elseif covid="34" and request("itemd")="3" then%>
<OPTION VALUE="33" SELECTED>CBP_iInfit
<% elseif covid="35" and request("personm")="1" then%>
<OPTION VALUE="35" SELECTED>CBP_G/G/M
 <% elseif covid="35" and request("personm")="2" then%>
<OPTION VALUE="35" SELECTED>CBP_G/G/Outfit
<% elseif covid="35" and request("personm")="3" then%>
<OPTION VALUE="35" SELECTED>CBP_G/G/Infit
 <%  
    end if  
 

if covid="12" then %>
 <OPTION VALUE="12" SELECTED>ANOVA
<% else %>
<OPTION VALUE="12">ANOVA
 <%  
    end if %>
 <% if covid="17" then %>
<OPTION VALUE="17" SELECTED >Summary
<% else %>
<OPTION VALUE="17">Summary
 <% end if %>
 <% if covid="27" then %>
   <OPTION VALUE="27"  SELECTED >Overall Fit
   <% else %>
    <OPTION VALUE="27" >Overall Fit
 <% end if %> 
  <OPTION VALUE="10">Cronban
     <OPTION VALUE="01">Person Forest
     <OPTION VALUE="02">Item Forest
 <% if covid="03" then %>
   <OPTION VALUE="03"  SELECTED >Wright Map
   <% else %>
    <OPTION VALUE="03" >Wright Map
 <% end if %>
 <% if covid="13" then %>
   <OPTION VALUE="13"  SELECTED >Wright Map(Groups)
   <% else %>
    <OPTION VALUE="13" >Wright Map(Groups)
 <% end if %>
<% if covid="04" then %>
    <OPTION VALUE="04" SELECTED  >KIDMAP
 <% else %>
 <OPTION VALUE="04">KIDMAP
 <% end if %>
  <% if covid="05" then %>
  <OPTION VALUE="05" SELECTED  >ICC_ca
 <% else %>
  <OPTION VALUE="05">ICC_ca
 <% end if %>
<% if covid="06" then %>
   <OPTION VALUE="06"  SELECTED>DIF(2 groups)
 <% else %>
   <OPTION VALUE="06">DIF(2 groups)
 <% end if %>
<% if covid="07" then %>
  <OPTION VALUE="07"  SELECTED>DIF(Graph)
 <% else %>
  <OPTION VALUE="07">DIF(Graph)
 <% end if %>
<% if covid="08" then %>
  <OPTION VALUE="08" SELECTED >Person Fit Plot(Outfit)
 <% else %>
 <OPTION VALUE="08">Person Fit Plot(Outfit)
 <% end if %>
<% if covid="20" then %>
  <OPTION VALUE="20" SELECTED >Person Fit Plot(Infit)
 <% else %>
 <OPTION VALUE="20">Person Fit Plot(Infit)
 <% end if %>

<% if covid="09" then %>
  <OPTION VALUE="09"  SELECTED>Simulation
 <% else %>
  <OPTION VALUE="09">Simulation
 <% end if %>

<% if covid="29" then %>
  <OPTION VALUE="29"  SELECTED>custermized Simu.
 <% else %>   
 <% end if %>

<% if covid="36" then %>
 <OPTION VALUE="36"  SELECTED>Item corr(Infit,delta) Cluster
 <% else %> 
 <OPTION VALUE="36" >Item corr(Infit,delta) Cluster
 <% end if %>
  <% if covid="37" then %>
 <OPTION VALUE="37"  SELECTED>KIDMAP (Z,delta) Cluster
 <% else %> 
 <OPTION VALUE="37" >KIDMAP (Z,delta) Cluster
 <% end if %>
 <% if covid="38" then %>
 <OPTION VALUE="38"  SELECTED>Item corr(delta,corr) Cluster
 <% else %> 
 <OPTION VALUE="38" >Item corr(delta,corr) 
 <% end if %>
 <% if covid="39" then %>
 <OPTION VALUE="39"  SELECTED>Item dist(delta,dist) Cluster
 <% else %> 
 <OPTION VALUE="39" >Item dist(delta,dist) Cluster
 <% end if %>
 <% if covid="40" then %>
 <OPTION VALUE="40"  SELECTED>Item(dist,corr) Cluster
 <% else %> 
 <OPTION VALUE="40" >Item(dist,corr) Cluster
 <% end if %>
 <% if covid="41" then %>
 <OPTION VALUE="41"  SELECTED>Slope Graph Contingency Outfit
 <% else %> 
 <OPTION VALUE="41" >Slope Graph Contingency Outfit
 <% end if %>

<% if covid="29" then %>
  <OPTION VALUE="11"  SELECTED>Kendall
 <% else %>   
 <% end if %>
 
    
 <% if covid="15" then %>
 <OPTION VALUE="15" SELECTED>Measure_Outfit
<% else %>
<OPTION VALUE="15">Measure_Outfit
 <% end if %>
    
 <% if covid="16" then %>
 <OPTION VALUE="16" SELECTED>Measure_Infit
<% else %>
<OPTION VALUE="16">Measure_Infit
 <% end if %> 

 <% if covid="18" then %>
<OPTION VALUE="18" SELECTED >Measure_Rawscore
<% else %>
<OPTION VALUE="18">Measure_Rawscore
 <% end if %>
 
</SELECT> <BR>
 
<font color=red>KIDMAP person#</font> 
       <input type="text"  name="kid" value="<%=kid%>"  style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size="4"><br> 
    Bubble Size<input type="text"  name="jsize" value="3"  style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size="4"><br>
  <input type="submit" value="Submit"  style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px"><br>   

<% if covid="03" or covid="13" or covid="05" then %>
   <font color=black>WrightMap(or ICC) dotted with dashes</font> <SELECT NAME="dash">
      <OPTION VALUE="No" >No
     <OPTION VALUE="Yes" SELECTED>Yes
       </SELECT> <BR> adjustwright
<% end if
   if covid="03" or covid="13"  then %> 
       Wright  move to left<input type="text"  name="adjustwright" value="0"  style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size="4"><br>
   <% end if %>
  <font color=black>Fit Types</font> <SELECT NAME="fitstat">
    <% if covid="20" or covid="03" or covid="13" then %>
    <OPTION VALUE="Infit" SELECTED>Infit MNSQ
       <OPTION VALUE="Outfit" >Outfit MNSQ
  <% else %>
      <OPTION VALUE="Outfit" SELECTED >Outfit MNSQ
      <OPTION VALUE="Infit" >Infit MNSQ
 <% end if %>
       </SELECT> <BR> 
 
 

<font color=red>ICC Item#</font>      
       <input type="text"  name="jitem" value="1"  style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size="4"><br>
    
<font color=red>Group#</font> <SELECT NAME="groupabc">    
     <OPTION VALUE="1" SELECTED>1
     <OPTION VALUE="2">2 
 <OPTION VALUE="3">3 
 <OPTION VALUE="4">4 
 <OPTION VALUE="5">5
</SELECT> 
<font color=red>Rasch</font> <SELECT NAME="raschabc">    
     <OPTION VALUE="1" SELECTED>Rasch
     <OPTION VALUE="2">2PL
</SELECT>
<% if covid="29" then %>
      <br>
   item difficulties<input type="text" name="delta1" style="background-color:lightgreen;" value="-3,-2,-1,0, 1, 2, 3 " size="20"><br> 
   threshold difficulties<input type="text" name="delta2" style="background-color:lightgreen;" value="-1,0,1" size="5">(abilities put in the column name)
<% end if %>
 
           <BR>
 

 <input type="button" name="button" value="R language" onClick="location.href='cbp.asp'" style="width:250px;font-
size:13pt;padding:2px; border:3px solid green"> <BR>

  

   <input type="button" name="button" value="Forest plot" onClick="location.href='../kpiall/forestplot.asp'" style="width:500px;font-size:13pt;padding:2px; border:3px solid green"><br>   
 
<input type="button" name="button"   value="Read me" onClick="location.href='../article/article16/Copy-PastingMethod.pdf'" style="width:500px;font-size:13pt;padding:2px; border:3px solid green"><BR>  
<input type="button" name="button"   value="Example0.txt in WINSTEPS" onClick="location.href='../article/article16/example0.txt'" style="width:500px;font-size:13pt;padding:2px; border:3px solid green">  
      
  </form>   
   </center> 
  

<%  response.end
   
   end if

     Session("content")=trim(request("content"))


  adjustwright=request("adjustwright")
      fitstat=request("fitstat")
      content= trim(request("content"))
 
         content2= request("content2") 
      content=replace(content,chr(10),"")
       content2=replace(content2,chr(10),"")
     contentaa2=split(content2,chr(13))
     contentaa=split(content,chr(13))


 
    content="":content2="":mswitch=0 
  pointintext=0
   for jk=0 to ubound(contentaa)
    if len(trim(contentaa(jk)))>0 then
       if (instr(contentaa(jk),",")>0 or instr(contentaa(jk),"	")>0)  then
         abcc=contentaa(jk)
         abcc=replace(abcc,"	",",")
          content=content & abcc & chr(13) 
        end if 
  
      if mswitch=0 then
        arr=split(abcc,",")

         redim itemname(ubound(arr)+1)
            
           for j=1 to ubound(arr)+1
               itemname(j)=arr(j-1)
           
           next
         mswitch=1 
      else
          arr=split(abcc,",")
   
         if isnumeric(arr(ubound(arr)))=false and jk>=1 then
             response.write "groups symble is not numeric at " & jk & " " & arr(ubound(arr))
             response.end
         end if  
      end if  
     end if

      if instr(contentaa(jk),".")>0 then
          pointintext=1
       end if
 
   next

  contentaa=split(content,chr(13))
 
 if instr(lcase(contentaa(0)),"name")=0 or instr(lcase(contentaa(0)),"group")=0 then
   response.write "No name or group in data head"
    response.end
end if
 if cdbl(logtrans)=4 and pointintext=1 then
      response.write "Errors in responses with points"
       response.end
 end if

  contentaa=split(content,chr(13))
 ' contentaa2=split(content2,chr(13))

 
    
  jsize=cdbl(request("jsize"))
 
  logtrans=request("logtrans")
 logtrans= cdbl(logtrans)

  if logtrans="" then logtrans=4
  categoryabc=request("categoryabc")
  if categoryabc="" then categoryabc=4
   if logtrans=3 then  categoryabc=1
   Groupabc=request("groupabc")
     if groupabc="" then groupabc=1


 'response.write personname


 
 if instr(content,";")>0 then
   content=replace(content,";",chr(13))
   contentaa=split(content,chr(13))
 else  
    contentaa=split(content,chr(13))
 end if 




 response.write ubound(contentaa)   & "=rows(including head labels)<br>"
   mcol=0:mrow=ubound(contentaa)+1
    if len(trim(contentaa(ubound(contentaa))))<2 then
        mrow=ubound(contentaa) 
    end if


   mrowerror=0
  for j=1 to ubound(contentaa)
     contentaa(j)=replace(contentaa(j),chr(13),"")
    contentaa(j)=replace(contentaa(j),chr(10),"")
    'response.write contentaa(j) & "<br>"
     arr=split(contentaa(j),",")
       if ubound(arr)>2 then
     mcol=mcol+ubound(arr)-2+1
     end if
  '  response.write ubound(arr)-2 &   "  "   &  mcol &   "  " & contentaa(j) & "<br>"
    
    if len(contentaa(j))<3 then
       mrowerror= mrowerror+1
     end if
 next 

  if ubound(contentaa)=1 then
       kano="YES" 
  end if  
 
kmnum=mrow
  
redim mpersonarr(kmnum)
 mrow=ubound(contentaa)+1- mrowerror


    
 content2=replace(content2,chr(10),"")
         groupabc=cdbl(groupabc)
     ngroupabc=1
     redim  conabc2(mrow)
       redim personname2(mrow)
  ' if ubound(contentaa2)+1<mrow then

            redim contentaa2(mrow-1)
          for jk=1 to mrow-1   'grom 1 to mrow
                 contentabb=split(contentaa(jk),",")
                abcc=contentabb(ubound(contentabb)) 'group
                               
                 personname2(jk)=contentabb(ubound(contentabb)-1) 'name
                 if trim(abcc)>"" then
                    contentaa2(jk)=mid(trim(abcc),groupabc,1)
                  
                 else
               
                            if jk<mrow/2 then
                               contentaa2(jk)=0 
                             else 
                                contentaa2(jk)=1 
                             end if
               end if
               
          next  
  ' end if
 

 if request("covid")<"03" then
   groupabc=""
 
end if  
 
 dash=request("dash")
 
            


    response.write "the number of persons=" & Lbound(contentaa2)+1 &" - " & mrow-1  &"<br>"
      num_q=sqr(mrow):max_group=0:min_group=6660



    for jk=2 to mrow
 
     if mrow  >ubound(contentaa2)+1  then
 
              if round(jk/num_q,0)> ngroupabc+1 then   
                  conabc2(jk-1)=round(jk/num_q,0)-1 
              else
                  conabc2(jk-1)=round(jk/num_q,0)
              end if   
     else   ' mrow=ubound(contentaa2)+1
 
       ' mrow=ubound(contentaa2)+1
        
          
    
    end if


     ' response.write jk & "aaa" & conabc2(jk) & " "&  ngroupabc & "<br>"
         if jk <=ubound(contentaa2)+1 then 
 if jk=47 then
      
        'response.write "AAAAA" &  contentaa(jk) 

        'response.end 
  end if 
          if  int(contentaa2(jk-1))> ngroupabc  then
            ngroupabc=int(contentaa2(jk-1))
          end if 

          end if       

       conabc2(jk-1)=Cdbl(contentaa2(jk-1))
         if conabc2(jk-1)>max_group then max_group=conabc2(jk-1)
           if conabc2(jk-1)<min_group then min_group=conabc2(jk-1) 

    next


   if mcol/(mrow-1)<>int(mcol/(mrow-1)) then
       response.write  mcol & " " &  mrow -1& " " & round(mcol/(mrow-1),2) & "error"
         for j=1 to len(content)
             if mid(content,j,1)<>"," then
               response.write j & " " & mid(content,j,1) & " " & asc(mid(content,j,1)) & "<br>"
             end if
         next
     
    end if
 
'==============================
 
 kid=cint(request("kid"))
  if kid<1 or kid>kmnum then
     response.write "Invalide Value" & kid
     response.end
   end if

 
  itemno=mcol/(mrow-1)
  

   
  mrow=mrow-1
  personno=mrow 
 kmax=0:kmin=999999

  redim personse(personno),personname(personno)
   redim test(mrow, itemno)
 
  redim personname(personno)
    allmean=0:numk=0
    for jk=1 to personno    'ubound(contentaa)
        arr=split(contentaa(jk),",")
         
         personname(jk)=arr(ubound(arr)-1)
   
           mean=0:sq=0
        for j=0 to ubound(arr)-2
            
              if  isnumeric(arr(j))=false then
                  arr(j)="."
                test(jk, j+1)=arr(j)                
              else      
                arr(j)=  CDbl(arr(j))  
                if arr(j)>kmax then kmax=arr(j)
                 if arr(j)<kmin then kmin=arr(j)
             
                    test(jk, j+1)=arr(j) 
                 mean=mean+arr(j) 
                 sq=sq + arr(j) ^2
               allmean=allmean+arr(j)
              end if
       
      next 
     

               if itemno>6 and personno>6 then
                   mean=mean/itemno

                   sd=sq/itemno-mean^2
                   personse(jk)=round(sd,4)
                     if sd=0 then
                         ' response.write "SD=0  " &  jk & " Col=" & ubound(arr)+1 & "row=" & personno &" not allow SD to be zero=" & sd & " mean=" &  mean & " sq=" & sq
                        '  response.end
                     end if
             end if
    next


       allmean=allmean/(personno*itemno)
  for j=1 to itemno
        mean=0:sq=0
    for jk=1 to personno


            if isnumeric(test(jk,j))=True then
              mean=mean+test(jk,j) 
              sq=sq + test(jk,j) ^2
            end if
  
    next
              if itemno>6 and personno>6 then
                mean=mean/personno
                   sd=sq/personno-mean^2
                     if sd=0 then
                        '  response.write "personno=" & personno &" Col=" & itemno &" not alow SD to be zero=" & sd & " mean=" &  mean & " sq=" & sq
                         ' response.end
                     end if
             end if
   
  next

 
   if mrow=ubound(contentaa2)+1 then
     response.write "By defaulted group labels=" &  ngroupabc+1 &"<br>"
  end if
if request("covid")="" then
%>
 
<table width="350" border="0" cellspacing="0" bgcolor="#FFFFFF">
 <tr><td>
<div align="center">
  <p>
<a href="https://pay.ecpay.com.tw/CreditPayment/FromCredit?c=3022346&Enn=e&installment=0&payOff=0 alt=English payment">
<img src="http://www.healthup.org.tw/paperabc/images/logo.png" width="100" height="100" border="1"></a></td>
<% if request("ecpayabc")>"" then %>
 <td><img src="http://www.healthup.org.tw/paperabc/ecpayabc.jpg" width="100" height="100" border="1"> </td>
 <tr><td><div align="center">
<a href="https://p.ecpay.com.tw/Opdh1"><img src="https://payment.ecpay.com.tw/Content/themes/WebStyle20170517/images/ecgo.png" width="100" height="100"/></a> 
   </td> <td>
  <a href="https://p.allpay.com.tw/Sw3oL"><img src="https://payment.allpay.com.tw/Content/themes/WebStyle201404/images/allpay.png" width="100" height="100"/></a>
   </td></tr>
 <% 
else %>
   <td><div align="center">
  <p>
<a href="http://www.healthup.org.tw/raschonline/" alt="PaperABC">
<img src="http://www.healthup.org.tw/paperabc/images/coding.png" width="100" height="100" border="1"></a>  </td></tr>
<% end if
 %></p> 
  
 </table>
<%   end if
    maxcat=-99999 
    mincat=99999
      for jk=1 to personno
              'response.write test(jk,j) &"aa " & mincat &   "bb " &"aa "& maxcat & ",<br>"
        for j=1 to itemno
         if isnumeric(test(jk,j)) then 
           if CDbl(test(jk,j))<0 then test(jk,j)=0
          if CDbl(test(jk,j))>=maxcat then
              maxcat=CDbl(test(jk,j))
                mjkk=jk
                mjkk2=j
           '     mname=mpersonarr(jk-1)
          end if

          if CDbl(test(jk,j))<=mincat then
              mincat=CDbl(test(jk,j))
            
          end if 
      end if
        next 
      next     
 
   '  response.write maxcat   &" " & mincat  &" " & mjkk &" " & mjkk2 & mname  &"<br>"
  

  redim item(itemno),person(personno) 

 
   'redim test(personno,itemno)
    redim  rawscore(itemno)
  redim  rawscore(itemno)

 mscore9=0  

   xmincat=0
  if maxcat<=100 and mincat>=0 and  maxcat>1 then
    '   mincat=0: maxcat=100*1.3 
     '  xmincat=1
  else
  '  mincat= mincat*.67: maxcat= maxcat*1.3 
    '   xmincat=1
  end if
 
 

 redim test2(personno,itemno),test3(personno,itemno),test4(personno,itemno)
 
  category_number= maxcat - mincat+1
  if categoryabc>category_number-1 then
   categoryabc=category_number-1
  end if
  response.write "category_number=" &  category_number & ", Max.=" & maxcat & ", Min.=" & mincat  &" Upper Cat.  number  Type(" & categoryabc & "=thresold)=logtrans" & logtrans & "=RSM<br>  "   
 simulatemincat=mincat
    simulaterange=(maxcat-mincat)


   if  mincat>0 then 
      mmaddn= mincat-0 
       maxcat= maxcat-mmaddn
        mincat=0
   else
      mmaddn=0
   end if 
 
 
   


  for jk=1 to personno 
         ' response.write  "<br>  "&   jk  & ":"      
        for j=1 to itemno
           if isnumeric(test(jk,j))=true then
              itemx="item" & j       
               test(jk,j)=test(jk,j)-mmaddn
               test2(jk,j)=test(jk,j) 
                test3(jk,j)=test(jk,j) 
                   test4(jk,j)=test(jk,j)
               
                 
                  test(jk,j)=round((test(jk,j)-mincat)/(simulaterange),2)
       
  

                 if logtrans=1 then 
                     test(jk,j)=round(log(test(jk,j)*round(exp(categoryabc),0)+1),0)              
                   ' test(jk,j)=round(log(test(jk,j)+1),0),0) 
                  elseif logtrans=2 then 
                     test(jk,j)=round(test(jk,j)*10/(10/categoryabc),0)
        
                       if test(jk,j)>categoryabc then  test(jk,j)=categoryabc 
                  elseif logtrans =3 then 
                      test(jk,j)=test(jk,j)   'continous scale
                  elseif logtrans =4 then 
                     test(jk,j)=test4(jk,j)
                  end if

           elseif categoryabc<4 then
                 test(jk,j)=0
           end if 'isnumeric
                    ' response.write trim(test(jk,j))  & "," 
         
        next 
        
  next
        
   
  
   
 maxcat=categoryabc
 mincat=0
   redim catcalibrate(maxcat-mincat+1)
 redim catobs(maxcat-mincat+1),catexp(maxcat-mincat+1),catresi(maxcat-mincat+1),catthresh(maxcat-mincat+1),catadj(maxcat-mincat+1)
 
 ' kendall'coefficient
   redim morder(itemno),morder2(itemno)
     for jk=1 to itemno
           morder(jk)=jk
           morder2(jk)=jk
      next

    for jk=0 to maxcat-mincat
             catcalibrate(jk)=0
           catobs(jk)=0
           catexp(jk)=0
     next 
  for jk=1 to personno
    For i = 1 To itemno-1  
         For j = i + 1 To  itemno              
             If test3(jk,i) > test3(jk,j) Then
  ' response.write   morder(i) & "aa " & test3(jk,i) & "bb " & test3(jk,i) & "<br>"     
                 SrtTemp = test3(jk,j)
                 test3(jk,j) = test3(jk,i)   
                 test3(jk,i) = SrtTemp 'from small to large         
                 SrtTemp2 =  morder(j)   
                  morder(j)   =  morder(i) 
                  morder(i)  = SrtTemp2   
              ' response.write   morder(i) & "aa " & test3(jk,i) & "bb " & test3(jk,i) & "<br>"        
             End If             
         Next 
     Next 

    
           for jm=1 to itemno 
                       kvalue=test3(jk,jm):mequal=jm:mcount=1
                  if jm<itemno then
                   for jm2=jm+1 to itemno 
                         if test3(jk,jm2)=kvalue then
                             mequal=mequal+jm2
                             mcount=mcount+1
                            ' kvalue=test4(jk,jm2) 
                         else
                                exit for
                         end if
                   next
                   ' response.write jm & " " & kvalue & " " &  mequal & " " & mcount & " "
                        for jm2=jm  to  jm+mcount-1
                           test4(jk,jm2)=  mequal/mcount 
                           ' response.write jm & " " & test4(jk,jm2) & " " & morder(jm) & "<br> "              
                        next                   
                          jm=jm +mcount-1   
                 
                 else
                     test4(jk,jm)=itemno
                      ' response.write jm & " " & test4(jk,jm2) & "<br> "  
                 end if  
              next
             ' response.end
         For i = 1 To itemno-1    
          For j = i + 1 To  itemno               
             If  morder(i)  >  morder(j)  Then
                 SrtTemp = morder(j) 
                 morder(j)  = morder(i) 
                 morder(i)  = SrtTemp 'from small to large
                 SrtTemp2 =  test4(jk,j)
                  test4(jk,j)   =  test4(jk,i)
                  test4(jk,i) = SrtTemp2             
             End If
             Next 
           Next 
   
  next


     

%> <div>Ranking</div>
<table><tr>
<%
' for jk=1 to personno
 '   For i = 1 To itemno 
   '  response.write "<td>" & test4(jk,i)   & "</td>"
   ' next
    '    response.write  "<tr>" 
 ' next
' response.write  "</table>" 




  redim kendal_p(itemno)
   for j=1 to itemno
        kendal_p(j)=0  
     for jk=1 to personno 
       kendal_p(j)=kendal_p(j) + test4(jk, j)
    next 
  next 

  dfk=itemno
  dfm=personno
      mean_w=0
     for j =1 to itemno
         mean_w= mean_w + kendal_p(j)
     next 
          mean_w= mean_w/itemno 
       kendall_w=0
     for j =1 to itemno
         kendall_w= kendall_w + (kendal_p(j)-mean_w)^2
     next 
     ' response.write  kendall_w
      ' response.end   
     kendall_w= 12* kendall_w/(dfm^2*(dfk^3-dfk)) 
     chi_kendall=round((dfk-1)*dfm* kendall_w,2)
      kendall_w=round( kendall_w,2) 
     df=dfk-1
     corr_k=round((dfm*kendall_w-1)/(dfm-1),2)
 
%>

<%
'===========================
  category_number =maxcat-mincat +1
  ' category_number =2 ' ഫ   F  ƫ ,2 I p  
     allscore9=0
redim raw_pz(personno), raw_iz(itemno)
redim raw_p2(personno)
  redim raw_p(personno), raw_i(itemno),se_p(personno), se_i(itemno),raw_p_avg(personno),raw_i_avg(itemno)
     
personmax2=0:personmin2=8888
 for jk=1 to personno
              mscore9=0:mscore92=0
 
        for j=1 to itemno
             
               if isnumeric(test(jk,j)) then
                  mscore9 = mscore9+test(jk,j)
                  mscore92 = mscore92+test2(jk,j) 'raw score
                  allscore9=allscore9+test(jk,j)
                  rawscore(j)=rawscore(j)+test2(jk,j)
                   catobs(test(jk,j))=catobs(test(jk,j))+1 
               end if             
         next
 
          raw_p(jk)=mscore9
           if raw_p(jk)<>maxcat*itemno and raw_p(jk)>personmax2 then
                    personmax2=raw_p(jk)
           end if
          if raw_p(jk)<>0 and raw_p(jk)<personmin2 then
                    personmin2=raw_p(jk)
           end if
          raw_p2(jk)=mscore92         
                raw_p_avg(jk)= raw_p(jk)/itemno          
                   raw_p(jk)=mscore9
                   raw_pz(jk)=mscore9  
          if jk=personno then exit for            
 next  
  


         allscore9_p= allscore9/itemno

  itemmax2=0:itemmin2=8888
  for j =1 to itemno
              mscore9=0:mscore92=0: raw_ia=0
        for jk=1 to personno
             
               if isnumeric(test(jk,j)) then
                  mscore9 = mscore9+test(jk,j)
                  mscore92 = mscore92+test2(jk,j) 'raw score
                  allscore9=allscore9+test(jk,j)
                  raw_ia=raw_ia+test2(jk,j)
                
                   
               end if             
         next 
                raw_i(j)=raw_ia
                    if raw_i(j)<>maxcat*personno and raw_i(j)>itemmax2 then
                      itemmax2=raw_i(j)
                   end if
           if raw_i(j)<>0 and raw_i(j)<itemmin2 then
                    itemmin2=raw_i(j)
           end if
               raw_iz(j)= raw_ia
               raw_i_avg(j)= raw_i(j)/personno 
          if j =itemno then exit for            
      next
       


  
      allscore9_i= allscore9/personno
  
       'PTMEASUR==PTMEASUR+(test(jk,j)-raw_p_avg(jk))*(raw_i(j)-allscore9_p)    
 redim expect(personno,itemno),Var(personno,itemno), Zscore(personno,itemno),residual(personno,itemno),kurtosis(personno,itemno),kurtosis2(personno,itemno)
   

     redim item_error(itemno),item_var(itemno)
    
 iterat=request("iterat")
 if iterat="" then iterat=40
  redim Aparameter(itemno)

   extremeshiftp=0: extremeshifti=0
 for iteration=1 to iterat
 for j=1 to itemno
   Aparameter(j)=1 
 next

          zscore_mean=0 :zscore_sd=0
       For CATa = 0 To categoryabc 'maxcat - mincat      
         catexp(CATa)=0
       Next 

        redim  item_var(itemno)
        redim item_error(itemno)
        resi_a=0: resi_b=0
     sumsqerror=0 
   redim  var_p(personno), person_exp(personno)    
   for jk=1 to personno
             person_error=0:person_max=-100: person_min=100
            sqaure_resi=0 

  perfect_i=0:perfect_p=0
          
      
     for j=1 to itemno
                 
     If IsNumeric(test(jk,j)) = True Then
     
   logit = person(jk) -item(j)
        
   normalizer = 0 ' this will force the sum of the probabilities = 1.0

   sumsqu = 0 
   currentlogit = 0: all_asum = 0
   ReDim expaaa(maxcat - mincat+1)
    catcalibrate(0)=0
   For CATa = 0 To  categoryabc 'maxcat - mincat
            msum_tau = 0
          if category_number>2 then
              For jk2 = 0 To CATa
                  msum_tau = msum_tau + catcalibrate(jk2)     
              Next 
           end if
           expaaa(CATa) = Aparameter(j)*Exp(CATa * logit - msum_tau)
      all_asum = all_asum +  expaaa(CATa)        
   Next 

    exp_a = 0: kurtosisZ = 0 
     For CATa = 0 To maxcat - mincat
       exp_a = exp_a + CATa * expaaa(CATa) / all_asum      
     Next 
        if logtrans<4 then
             var_a =exp( logit)/(1+exp( logit))             
         end if



       var_a = 0
       For CATa = 0 To maxcat - mincat
        kurtosisZ = kurtosisZ + (exp_a - CATa) ^ 4 * expaaa(CATa) / all_asum
        var_a = var_a + (exp_a - CATa) ^ 2 * expaaa(CATa) / all_asum
        catexp(CATa) = catexp(CATa) + expaaa(CATa) / all_asum
    
      Next 
         if logtrans<4 then
             var_a =exp_a *(1-exp_a )             
         end if
 
             EXPECT(jk, j) = exp_a
             var(jk, j) = var_a
           
             residual(jk, j)  = test(jk,j) - exp_a
              person_error=person_error+residual(jk,j) 
              var_p(jk)=var_p(jk)+var_a  
              item_var(j)=item_var(j)+var_a 
           if person(jk)>0 then
             Aparameter(j)=Aparameter(j)+ residual(jk, j)/personno 
           end if
     
              item_error(j)=item_error(j)+residual(jk,j)
                    if var(jk,j)>0 then
                            Zscore(jk, j)  =  residual(jk,j)/var(jk,j)^0.5 
                                     kurtosis(jk, j) = kurtosisZ /Var(jk, j)^2  'outfit
                                  kurtosis2(jk, j) = kurtosisZ -Var(jk, j)^2   'infit
                    else
                         Zscore(jk, j) =0
                          kurtosis(jk, j) = 1  'outfit
                              kurtosis2(jk, j) = 1   'infit
                     end if
              sqaure_resi= sqaure_resi + residual(jk,j)^2
              zscore_mean=zscore_mean+ Zscore(jk, j) 
                zscore_sd= zscore_sd+Zscore(jk, j)^2 

                 if raw_i(J)=personno*maxcat then
                     kurtosis2(jk, j) =0.00001
                     perfect_i=perfect_i+1
                 end if
               if raw_p(jk)=itemno*maxcat then
                     kurtosis2(jk, j) =0.00001
                      perfect_p=perfect_p+1
                 end if
 
    else 'missing
        
            
             var(jk, j) =  "."
             residual(jk, j)  =  "."
             Zscore(jk, j)  =  "."
             kurtosis(jk, j) =  "."
             kurtosis2(jk, j) =  "."
    end if
      
      next 'item
           resi_a= resi_a+person_error 
      
           
      if abs(person_error)>sumsqerror then
         sumsqerror= abs(person_error) 
      end if

         if var_p(jk)<0.0001 then var_p(jk)=0.001
         
           if var_p(jk)>0 then       
           if person_error/var_p(jk)>10 then
                person(jk)=10'  6.07
          elseif person_error/var_p(jk)<-10 then
                   person(jk)=-10 ' -6.07
          else
            person(jk)=person(jk)+  person_error/var_p(jk) 
             if  person(jk)>10 then person(jk)=10
             if  person(jk)< -10 then person(jk)=-10

       '   response.write "<br>" & round(person(jk),2) & " " & round(person_error,2) & " " & round(var_P(jk),2) & " " & round(person_error/var_p(jk),2) & "=======<br>" 
          end if
                 person_exp(jk)=  sqaure_resi/var_p(jk)  'outfit 
            else
                  person(jk)=person(jk) 
                  person_exp(jk)=  1  'outfit  
             end if
        if person_exp(jk)<.0016 then person_exp(jk)=0.0016
         if raw_p(jk)=personmax2 then 
           personmax2m=person(jk):pvarmax2m=var_p(jk)
            '  response.write  raw_p(jk) & "<br>"
          ' response.write var_p(jk) & "<br>"
          ' response.write person(jk) & "<br>"
         
         end if
         if raw_p(jk)=personmin2 then 
            personmin2m=person(jk):pvarmin2m=var_p(jk) 
            '  response.write  raw_p(jk) & "dd<br>"
          ' response.write var_p(jk) & "dd<br>"
          ' response.write person(jk) & "dd<br>"
          end if 
         if person(jk)=10 or person(jk)=-10 then extremeshiftp=1
   next 'person
       
           
               item_avg=0
          for j=1 to itemno
           if abs(item_error(j))>sumsqerror then  
             '  sumsqerror= abs(item_error(j))
                          
            end if
                resi_b=resi_b+ item_error(j)
                 
            if item_var(j)<0.0001 then item_var(j)=0.001
            
         if item_var(j)>0 then
            if item_error(j)/item_var(j)>10 then
              item(j)= 10
            elseif item_error(j)/item_var(j)<-10 then
              item(j)= -10
            else
                 item(j)=item(j)- item_error(j)/item_var(j) 
             if   item(j)> 10 then  item(j)=10
             if   item(j)< -10 then  item(j)=-10
            end if
           ' response.write  "<br>" & j &  " " & round(item_error(j),2) & ":" & round(item_var(j) ,2)
            
          else
              item(j)=item(j)
           end if
                item_avg=item_avg+item(j)
 
            if raw_i(j)=itemmax2 then itemmax2m=item(j):ivarmax2m=item_var(j)
            if raw_i(j)=itemmin2 then itemmin2m=item(j):ivarmin2m=item_var(j) 
              if item(j)=10 or item(j)=-10 then extremeshifti=1
          next
  
                item_avg= item_avg/itemno
          for j=1 to itemno
              item(j)= item(j)-item_avg
          '  response.write  "<br>item" & j &  " " & round(item(j),2) & ":" & round(item_avg,2)
           next 
           cat_avg=0

          For jkm = 0 To category_number - 1 ' mincat
               'catexp
                catresi(jkm) = catobs(jkm)-CATEXP(jkm)
         ' response.write "<br>cat" & round(catobs(jkm),2) & " " & round(CATEXP(jkm),2) & " " & round(catresi(jkm),2) & "<br>"

               if abs(catresi(jkm))>sumsqerror then  
                   sumsqerror= abs(catresi(jkm))
               end if

              if jkm>0 then

                 if catobs(jkm)>0 and catexp(jkm)>0 then
                   if catobs(jkm-1)/catobs(jkm)>0 and catexp(jkm-1)>0 then
                     catthresh(jkm) =catcalibrate(jkm)+Log(catobs(jkm-1)/catobs(jkm)) -LOG(catexp(jkm-1)/catexp(jkm))
                   end if
                  else
                  catthresh(jkm) =catcalibrate(jkm)
                 

                 end if  
              cat_avg=cat_avg+catthresh(jkm)
              else
                  catthresh(jkm) =0
              end if   
 
          Next 
 
         
              if (category_number - 1)>0 then
                 cat_avg= cat_avg/(category_number - 1)
              else
                  cat_avg=0.001
              end if

            catadj(0)=0
          For jkm = 1 To category_number - 1 ' mincat
                catadj(jkm)=catthresh(jkm)-cat_avg
              catcalibrate(jkm)=catadj(jkm)
    
          next 
       if sumsqerror<0.05 or  abs(sumsqerror2 - sumsqerror)<0.01 then
       response.write "Residuals=" & round(sumsqerror,2)  & "previous vs. after= " & round(sumsqerror2,2)
            
            if sumsqerror2>0 and  sumsqerror>0 and iteration>1 then

              exit for
             end if
       end if
         sumsqerror2 = sumsqerror
     ' response.write "<font color=red>" & iteration & "</font>" & resi_a & " " & resi_b & " " & sumsqerror2 & " <br>"
     if iteration=24  then
      
     end if
     
  next 'iteration
  



   if logtrans<4 then 
response.write   "<br>"
response.write personno & " " & itemno  &  "<br>"

  for jk=1 to personno
     response.write jk & " " & round(person(jk),2) & "<br>"
  next

 for j=1 to itemno
     response.write j & " " & round(item(j),2) & "<br>"
  next
 

 response.write " sumsqerror2:" & round(sumsqerror2,2) & " sumsqerror:" & round(sumsqerror,2)
   'response.end


    end if



    if request("raschabc")="2" then
      ' for  j=1 to itemno           
             '   Aparameter(j)=round(Aparameter(j),2)                 
      ' next 
     end if

      adj_rate=1.5*1.5

          ' for extreme scores
      if extremeshiftp=1 then
       extremeperson=(personmax2m-personmin2m)/(personmax2-personmin2)*(maxcat*itemno-personmax2)*adj_rate+personmax2m
      extremeperson0=(personmax2m-personmin2m)/(personmax2-personmin2)*(0-personmin2)*adj_rate+personmin2m
      if (pvarmax2m-pvarmin2m)>0 then  
      vextremeperson=(personmax2m-personmin2m)/(pvarmax2m-pvarmin2m)*(extremeperson-personmax2m)+pvarmax2m
      vextremeperson0=(personmax2m-personmin2m)/(pvarmax2m-pvarmin2m)*(extremeperson0-personmin2m)+pvarmin2m
     else
     vextremeperson=.001
      vextremeperson0=.001
     end if
      vextremeperson= vextremeperson 
      vextremeperson0= vextremeperson0 
      if  vextremeperson0<0 then  vextremeperson0=1.84*1.84 
       if  vextremeperson<0 then  vextremeperson=1.84*1.84 


    for jk=1 to personno
        if person(jk)=10  then person(jk)=extremeperson: var_p(jk)=vextremeperson
        if person(jk)=-10 then person(jk)=extremeperson0: var_p(jk)=vextremeperson0
         if var_p(jk)<=0 then var_p(jk)=0.01 

     next

      end if



      if extremeshifti=1 then      
        extremeitem=(itemmax2m-itemmin2m)/(itemmax2-itemmin2)*(maxcat*personno-itemmax2)*adj_rate+itemmax2m
        extremeitem0=(itemmax2m-itemmin2m)/(itemmax2-itemmin2)*(0-itemmin2)*adj_rate+itemmin2m
        vextremeitem=(itemmax2m-itemmin2m)/(itemmax2-itemmin2)*(extremeitem-itemmax2m)+ivarmax2m
        vextremeitem0=(itemmax2m-itemmin2m)/(itemmax2-itemmin2)*(extremeitem0-itemmin2m)+ivarmin2m
         vextremeitem=vextremeitem
          vextremeitem0=vextremeitem0 

        if   vextremeitem0<0 then   vextremeitem0=1.84*1.84 
        if vextremeitem<0 then  vextremeitem=1.84*1.84 



        for j=1 to itemno 

        if item(j)=10 then item(j)=extremeitem: item_var(j)=vextremeitem
 
        if item(j)=-10 then
 
           item(j)=extremeitem0:item_var(j)= vextremeitem0
        end if
 
            if item_var(j)<=0 then item_var(j)=0.01

  
      next

      end if 
 
 

 '============================end iteration 

        For jkm = 1 To category_number - 1 ' mincat
                catadj(jkm)=catthresh(jkm)-cat_avg
              catcalibrate(jkm)=catadj(jkm)
         response.write "Steps..."  & JKM  & " difficulty=" & round(catcalibrate(jkm),2) &"<br>"
          next
        response.write "iteration...<br>" &  iteration & "<br>"

 'data point========================================================
  redim personxx(personno)
  redim personrk(personno)
  rktext="" : mtxt=""
   redim cutting(6) 
  
 for jk=1 to personno
                  if person(jk)>person_max then person_max=person(jk)
                  if person(jk)<person_min then person_min=person(jk)
             personxx(jk)=raw_p(jk)
  
                  kidat=round((6-person(jk))*2,0)
               if kidat<0 then kidat=0
               if kidat>25 then kidat=25  '3.5, 1 , -1.5, -4
  
                 ranking=Fix(kidat/5)+1
                  if ranking>5 then ranking=5
                 alpheta="ABCDEFGHIJK"
                 remainer=kidat mod 5
                   mtxt=mtxt & ranking

                rk=mid("ABCDEFGJIJK",ranking,1)  ' & right("0" & (remainer+1),2) 
               
                personrk(jk)=rk

           if instr(rktext, rk)=0 then
                rktext=rktext & rk
           end if

               cutting(ranking)=cutting(ranking) +round(person(jk),2)
   

 next 

      rktext=".........."
   for jk=1 to len(mtxt)
         rk=cdbl(mid(mtxt,jk,1))
          abc=mid("ABCDEFGJIJK",rk,1)
            if rk=1 then rktext=abc & mid(rktext, 2,len(rktext)-1)
            if rk=2 then rktext=  mid(rktext, 1,1) & abc & mid(rktext, 3,len(rktext)-2)
            if rk=3 then rktext= mid(rktext, 1,2) & abc & mid(rktext, 4,len(rktext)-3)
            if rk=4 then rktext= mid(rktext, 1,3) & abc & mid(rktext, 5,len(rktext)-4)
            if rk=5 then rktext= mid(rktext, 1,4) & abc & mid(rktext, 6,len(rktext)-5)
    next 
   rktext=trim(replace(rktext,".",""))
  
     krange2=person_max - person_min

  krange= len(rktext)  '(personno/6) '(person_max - person_min)/initial_g  ' defined itt groups initital
   
    jitem=request("jitem") 
  
  redim group_p(krange,4),group_p2(krange,4) 
  redim chsq_item(itemno),chsq_itemabc(itemno),chsq_itemabc2(itemno),chsq_itemabc3(itemno)
     chsq_jitem=0                 
    for j=1 to krange
       group_p(j,2)=0  'raw score
       group_p(j,0)=0  'count
        group_p(j,3)=0  'expect
          group_p(j,4)=0  'var      
    next
   
  
 
   tvalue=0:tvalue2=0:chsq1=0:chsq2=0:chsq3=0  
  for j=1 to itemno    
     for kangexx=1 to  krange 
       chsqstratabc=0      
           For i = 1 To  personno 

              if  isnumeric(test(i,j))=true then                                            
              if mid(rktext,kangexx,1)=  personrk(i)  then
                   group_p(kangexx,2)= group_p(kangexx,2)+test(i,j)  'raw score
                   group_p(kangexx,0)= group_p(kangexx,0)+1  'count
                    group_p(kangexx,3)=group_p(kangexx,3)+expect(i,j)  'expect
                     group_p(kangexx,4)=group_p(kangexx,4)+var(i,j)  'var 
              end if




 
              if mid(rktext,kangexx,1)=  personrk(i) and cdbl(jitem)=j  then
                   group_p2(kangexx,2)= group_p2(kangexx,2)+test(i,j)  'raw score
                   group_p2(kangexx,0)= group_p2(kangexx,0)+1  'count
                    group_p2(kangexx,3)=group_p2(kangexx,3)+expect(i,j)  'expect
                     group_p2(kangexx,4)=group_p2(kangexx,4)+var(i,j)  'var 
 
              end if
             end if 
  
          
          next


                 if  group_p(kangexx,4)=0 then  group_p(kangexx,4)=0.01
                 chsq_item(j) =chsq_item(j)+(group_p(kangexx,2)- group_p(kangexx,3))^2/group_p(kangexx,4) 

                if group_p(kangexx,3)>0 then

                  chsq_itemabc(j) =chsq_item(j)+ group_p(kangexx,2)*((group_p(kangexx,2)/group_p(kangexx,3))^1-1)
        chsq_itemabc2(j) =chsq_itemabc2(j)+ group_p(kangexx,2)*((group_p(kangexx,2)/group_p(kangexx,3))^0.01-1)
                  chsq_itemabc3(j) =chsq_itemabc3(j)+ group_p(kangexx,2)*((group_p(kangexx,2)/group_p(kangexx,3))^0.67-1)
                 end if
 

           if cdbl(jitem)=j then 
                if  group_p2(kangexx,4)=0 then  group_p2(kangexx,4)=0.01
                tvalue2= tvalue2+(group_p2(kangexx,2)- group_p2(kangexx,3))^2/group_p2(kangexx,4)   
             end if

     next  'for kangexx=1

         tvalue=tvalue+ chsq_item(j)
         tvalue2=tvalue2 

          chsq1=chsq1 + 2/(1*(1+1))*chsq_itemabc(j)
         chsq2=chsq2 + 2/(0.01*(0.01+1))*chsq_itemabc2(j) 
          chsq3=chsq3 + 2/(0.67*(0.67+1))*chsq_itemabc3(j) 
 
     next 'itemno

chsq3=abs(chsq3)
chsq1=abs(chsq1)
chsq2=abs(chsq2)
 
         'https://jsdajournal.springeropen.com/articles/10.1186/s40488-020-00108-7 
 
  response.write  "<table><tr><td> Strata  item= All items</td></td></tr></table>" 
   
             response.write  "<table><tr><td>Strata</td><td><font color=red>Sum</font></td><td>Count*L</td><td>Mean</td><td><font color=red>Expected</font></td><td>Variance</td></tr>" 

  for j=1 to krange
          if  mid(rktext,j,1)="A" then
                 mlogit=">3.5"    '3.5, 1 , -1.5, -4
          elseif mid(rktext,j,1)="B" then
                mlogit=">1.0"
          elseif mid(rktext,j,1)="C" then
                 mlogit=">-1.5"     
          elseif mid(rktext,j,1)="D" then
                 mlogit=">-4.0"
          else
                  mlogit="<=-4.0"
          end if     

      response.write "<tr><td>" & mid(rktext,j,1)  &"_" &   j & "(" &  mlogit  & ")</td><td>" & round(group_p(j,2),2) & "</td><td>" & group_p(j,0) & "</td><td align=right>" & round(group_p(j,2)/group_p(j,0),2) & "</td><td align=right>" & round(group_p(j,3),2) & "</td><td align=right>" & round(group_p(j,4),2) & "</td></tr>"
         
     next
  

       df= (krange -1)*itemno
     
           if  df >100 then
                df=100
             else
                  if tvalue>49 then
                       tvalue=round(tvalue/(tvalue/50),2)
                       df=round(df/(tvalue/50),0)
                       if tvalue>49.8 then tvalue=49.8
                  end if
             
         strSQL =  "Select * From chiquest where a0=" & round(tvalue,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
      


           if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df 
  
                   pro=round(objrsst(itema),3)
                 end if
              end if
           end if	
        if pro>1 then pro=1
 response.write "<tr><td>ChSQ=</td><td>" & round(tvalue,2) & "</td><td>df=(n*(k-1))</td><td align=right>" & df & "</td><td align=right><font color=red>prob.=</td><td align=right>" & round(pro,2) & "</font></td></tr>"
   response.write "<tr><td><a href=https://jsdajournal.springeropen.com/articles/10.1186/s40488-020-00108-7>Ref. in Eq 4(<font color=red>click</font>)</a></td><td></td><td></td><td align=right></td><td align=right></td><td align=right></font></td></tr>"
        
       response.write "</table>" 
   'Overall chsq=============================================================

 
response.write  "<table><tr><td> Strata_raw score item=" & jitem & "</td></tr></table>" 
   
             response.write  "<table><tr><td>Strata</td><td><font color=red>Sum</font></td><td> n </td><td>Mean</td><td><font color=red>Expected</font></td><td>Variance</td></tr>" 


  for j=1 to krange
        if group_p2(j,0)>0 then
      if  mid(rktext,j,1)="A" then
                 mlogit=">3.5"    '3.5, 1 , -1.5, -4
          elseif mid(rktext,j,1)="B" then
                mlogit=">1.0"
          elseif mid(rktext,j,1)="C" then
                 mlogit=">-1.5"     
          elseif mid(rktext,j,1)="D" then
                 mlogit=">-4.0"
          else
                  mlogit="<=-4.0"
          end if 

         response.write "<tr><td>" & mid(rktext,j,1)  &"_" &   j & "(" &  mlogit  & ")</td><td>" & round(group_p2(j,2),2) & "</td><td>" & group_p2(j,0) & "</td><td align=right>" & round(group_p2(j,2)/group_p2(j,0),2) & "</td><td align=right>" & round(group_p2(j,3),2) & "</td><td align=right>" & round(group_p2(j,4),2) & "</td></tr>"
        end if
     next


       df= (krange -1)*1
     
           if  df >100 then
                df=100
             else
                  if tvalue>49 then
                  '   tvalue=round(tvalue2/(tvalue2/50),2)
                  '    df=round(df/(tvalue2/50),2)
                  end if
        
         strSQL =  "Select * From chiquest where a0=" & round(tvalue2,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
      


           if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df 
  
                   pro=round(objrsst(itema),3)
                 end if
              end if
           end if	
if pro>1 then pro=1
 response.write "<tr><td>ChSQ=</td><td>" & round(tvalue2,2) & "</td><td>df=n(k-1)=</td><td align=right>" & df & "</td><td align=right><font color=red>prob.=</td><td align=right>" & round(pro,2) & "</font></td></tr>"
   jitemtxt="ChSQ=" & round(tvalue2,2) & "df=" & df & "prob.=" & round(pro,2)

 response.write "<tr><td><a href=https://jsdajournal.springeropen.com/articles/10.1186/s40488-020-00108-7>Ref. in Eq 4(<font color=red>click</font>)</a></td><td></td><td></td><td align=right></td><td align=right></td><td align=right></font></td></tr>"
        
       response.write "</table>" 
 response.write  "<table><tr><td>chSQ1 ==================================</td></tr></table>"
 response.write  "<table><tr><td> Strata(Pearson 1900) for All items</td></td></tr></table>" 
   
             
       df= (krange -1)*itemno
     
           if  df >100 then
                df=100
             else
                  if tvalue>49 then
                  '   tvalue=round(tvalue/(tvalue/50),2)
                  '    df=round(df/(tvalue/50),2)
                  end if
        
         strSQL =  "Select * From chiquest where a0=" & round(chsq1,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
      


           if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df 
  
                   pro=round(objrsst(itema),3)
                 end if
              end if
           end if	
        if pro>1 then pro=1
 response.write "<Table><tr><td>ChSQ=</td><td>" & round(chsq1,2) & "</td><td>df=k-1=</td><td align=right>" & df & "</td><td align=right><font color=red>prob.=</td><td align=right>" & round(pro,2) & "</font></td></tr></Table>"
   response.write "</table><table><tr><td>In pp171 in Invariant Measurement(George Engelhard , Jr.)</td><td></td><td></td><td align=right></td><td align=right></td><td align=right></font></td></tr>"
        
      response.write "</table>" 
 response.write  "<table><tr><td>chSQ2 ===================================</td></tr></table>"
  response.write  "<table><tr><td> Strata(Wilks,1935) for All items</td></td></tr></table>" 
   
            
       df= (krange -1)*itemno
     
           if  df >100 then
                df=100
             else
                  if tvalue>49 then
                  '   tvalue=round(tvalue/(tvalue/50),2)
                  '    df=round(df/(tvalue/50),2)
                  end if
        
         strSQL =  "Select * From chiquest where a0=" & round(chsq2,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
      


           if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df 
  
                   pro=round(objrsst(itema),3)
                 end if
              end if
           end if	
        if pro>1 then pro=1
 response.write "<Table><tr><td>ChSQ=</td><td>" & round(chsq2,2) & "</td><td>df=</td><td align=right>" & df & "</td><td align=right><font color=red>prob.=</td><td align=right>" & round(pro,2) & "</font></td></tr></Table>"
   response.write "</table><table><tr><td>In pp171 in Invariant Measurement(George Engelhard , Jr.)</td><td></td><td></td><td align=right></td><td align=right></td><td align=right></font></td></tr>"
        
       response.write "</table>"
 response.write  "<table><tr><td>chSQ3 ====================================</td></tr></table>"
  response.write  "<table><tr><td> Strata(Cressie & Read,1988) for All items</td></td></tr></table>" 
   

       df= (krange -1)*itemno
     
           if  df >100 then
                df=100
             else
                  if tvalue>49 then
                  '   tvalue=round(tvalue/(tvalue/50),2)
                  '    df=round(df/(tvalue/50),2)
                  end if
        
         strSQL =  "Select * From chiquest where a0=" & round(chsq3,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
      


           if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df 
  
                   pro=round(objrsst(itema),3)
                 end if
              end if
           end if	
        if pro>1 then pro=1
 response.write "<Table><tr><td>ChSQ=</td><td>" & round(chsq3,2) & "</td><td>df=</td><td align=right>" & df & "</td><td align=right><font color=red>prob.=</td><td align=right>" & round(pro,2) & "</font></td></tr></Table>"
   response.write "</table><table><tr><td>In pp171 in Invariant Measurement(George Engelhard , Jr.)</td><td></td><td></td><td align=right></td><td align=right></td><td align=right></font></td></tr>"
        
       response.write "</table>"
 response.write  "<table><tr><td>End ChSQ ================================================================</td></tr></table>"

 

  

       if request("covid")="27" then 
           response.end
       end if
   
     zzll=0
 
    for jk=1 to personno
         if krange2/krange>0 then 
          zgroup = int(i/(krange*personno/6))+1 
         else
            zgroup  =1
          end if
           ' response.write zgroup & "  " & krange2 & "  " & krange & "<br>"
         
          if zgroup<1 then
              zgroup=1
          end if
           if zgroup>= krange+1 then zgroup=krange   
          
           if raw_p(jk)>= group_p(zgroup,2) then
                 group_p(zgroup,1)= person(jk)
           end if
            
 ' response.write round(person(jk),2)  & "  " & raw_p(jk)  & "  " & zgroup & " " &  group_p(zgroup,0) & "  " &  group_p(zgroup,1)  & "  " &   group_p(zgroup,2) & "<br>"
      
     next
 
     ' response.write "iteration...<br>"
   ' do while  true
       mnum=0
 response.write "Group...strata=" &  krange & "<br>"
  response.write  "Category number=" & category_number & "<br>  "   
   
 
 'for j=1 to krange
 'response.write j & " " & group_p(j  ,2) & "d<br>"    
 'next
 'response.end
      ' if  mnum=0 then exit do  
  ' loop


 
  'calculate prob on responses
  redim outfitabc(personno),chisuareabc2(personno),infitabc(personno) 

  redim test_prob(personno+1,itemno),person_catprob(personno)

     person_catprobabc=0 
   for jk=1 to personno 
 
          person_catprob2=0
          outfit=0:chisuareabc=0
 for j=1 to itemno
    logit = person(jk) -item(j)
    currentlogit = 0: all_asum = 0

   ReDim expaaa(maxcat - mincat +1)
 
   For CATa = 0 To   maxcat - mincat
            msum_tau = 0 
       expaaa(CATa) = Exp(CATa * logit - msum_tau)
      all_asum = all_asum +  expaaa(CATa)        
   Next 
 
         obs_score= test(jk,j) 
        ' chisuareabc=chisuareabc+(abs(expaaa(1) / all_asum -test(jk,j))*100)^2/(100*expaaa(1) / all_asum)
          if isnumeric(test(jk,j))=true then  
            chisuareabc=chisuareabc+zscore(jk,j)^2
            prob_score =   abs(expaaa(1) / all_asum -test(jk,j))*10
            prob_score =  exp(logit-prob_score)/(1+exp(logit-prob_score))
               prob_score = 1-abs(expaaa(1) / all_asum - prob_score)   
           if prob_score >=0.99 then prob_score =0.99
         if prob_score <=0.01 then prob_score =0.01
     '  response.write round(expaaa(1) / all_asum,2) & "  " & test(jk,j)  & " " & round(prob_score,2) &"<br>"

    test_prob(jk,j)=prob_score/(1-prob_score) 'log(prob_score)  
          
 
       prob_cat_sum=prob_cat_sum  + prob_score/(1-prob_score) 'log(prob_score) 
       person_catprob2=person_catprob2 + prob_score/(1-prob_score) 'log(prob_score)
 
          outfit=outfit +zscore(jk,j)^2
         test_prob(personno+1,j)=test_prob(personno+1,j)+ test_prob(jk,j)
 
      end if 'true


next 'itemno

        person_catprob(jk)= person_catprob2'  (-2*person_catprob2)' 1/(-2*person_catprob2)
        person_catprobabc=person_catprobabc+person_catprob(jk)
         
             outfitabc(jk)= outfit/itemno
            
            strSQL =  "Select * From chiquest where a0=" & round(chisuareabc,1) & " and a2<2 "
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
              if objrsst.eof then
                  pro=0
              else
                 if gendernumber-1>100 or gendernumber-1<0 then
                    pro=1
                 else   
                   itema="A" &  itemno-1   
                   pro=round(objrsst(itema),3)
                 end if
              end if
      	
         chisuareabc2(jk)= round(pro,2)
       if person_catprob2>-0.05 then 
           person_catprob2=-0.05
      end if
     
                chi_mean=chi_mean+person_catprob2
       'response.write jk & " " & person_catprob(jk) & "<br>"
   next 'personno
       chi_probcat=-2*prob_cat_sum
       chi_mean=round(chi_mean/personno,0)
 dff=personno*itemno - (personno+itemno-1+(category_number-2*1))


'1 ===================================================

        ' free parameters are 74 + 25 - 1 + (3-category rating scale - 2 x 1 rating scale) = 99 
 
 'response.write  "#, expected count, total count, -2log(outfit)"
     redim cr(krange,2), group_score(krange)
    pattern_prob=0:cr_count=0
 
       for jk=1 to personno
           ' if raw_p(jk)>0 and raw_p(jk)<>allscoreuper then
                    
                for j=1 to krange  

                     if j<krange then
                         criterio=group_p(j+1,2) 
                      else
                          criterio=300000
                      end if  
                     if raw_p(jk)>=group_p(j,2) and raw_p(jk)< criterio then
                    
                        zgroup=j 
                          exit for
                     end if
                next
           
                 if zgroup>krange then zgroup=krange 
                 cr(zgroup,0)= cr(zgroup,0)+1
         
               percentabc=1- outfitabc(jk)/10 
             if  percentabc>1 then  percentabc=.99
             if  percentabc<=0 then  percentabc=.01
         
                 percentabc=log(percentabc)
 
         if percentabc>-0.01 then 
            percentabc=-0.01
         end if
           
                cr(zgroup,1)= cr(zgroup,1)+ 1/(-2*percentabc)  'https://www.rasch.org/rmt/rmt34a.htm       
                pattern_prob= pattern_prob +  1/(-2*percentabc)
                
                 cr_count=cr_count+1
         '   end if 
    'response.write round(person(jk),2) & "  " &  cr(zgroup,0) & "  " & group_score(zgroup)  & "  " &  cr(zgroup,2) & "<br>"
             
        next'personno
  
 zzzz=0:zzzz2=0
 for jk=1 to krange
            zgroup=jk
      
        if group_p(zgroup,0)>0 and pattern_prob>0 then
           cr(zgroup,2)=   cr_count* cr(zgroup,1)/pattern_prob 'expected count     
       end if

     'response.write   jk   & " " & round(cr(zgroup,2),4) & " " & cr_count & " "    &    round(cr(zgroup,1)/pattern_prob,4) & "<br>"           
    
  next

      
   
    'response.write "==================="  & zzzz & " bb" &  cr_count & "   " & zzzz2  & "<br>"

     chi_fit=0: g_two=0: CR3=0:cr_count=0
    for j =1 to  krange
          if group_p(j,0) and cr(j,2)>0   then

               chi_fit=chi_fit + cr(j,0)*((cr(j,0)/cr(j,2))-1) 
               g_two= g_two +  cr(j,0)*((cr(j,0)/cr(j,2))^0.001-1) 
                CR3=CR3+  cr(j,0)*((cr(j,0)/cr(j,2))^0.67-1) 
               cr_count=cr_count+1
     
         end if
 
    next

'2 ========================================
    redim cr(krange,2), group_score(krange)
    pattern_prob=0:cr_count=0
        redim person_gp(personno)        
   
       for jk=1 to personno
                person_gp(jk)=contentaa2(jk)
            
          '  if raw_p(jk)>=0 and raw_p(jk)<>allscoreuper then
                
               for j=1 to krange 
                     if j<krange then
                         criterio=group_p(j+1,2) 
                      else
                          criterio=300000
                      end if  
                     if raw_p(jk)>=group_p(j,2) and raw_p(jk)< criterio then
                        zgroup=j 
                      
                          exit for
                     end if
                next
                 if zgroup>krange then                  
                      zgroup=krange 
                 end if
                  cr(zgroup,0)= cr(zgroup,0)+1
                   
                cr(zgroup,1)= cr(zgroup,1)+ person_catprob(jk)   'https://www.rasch.org/rmt/rmt34a.htm 
                                        
                pattern_prob= pattern_prob +  person_catprob(jk) 
  'response.write  zgroup & " " &  cr(zgroup,0) & " "&  raw_p(jk)  & " "& round(group_p(zgroup,2),2) & " "& round(cr(zgroup,1),2) & " " & round(pattern_prob,2) &"<br>"  
                     if  cr(zgroup,1)=0 then  cr(zgroup,1)=0.0001 
                   cr_count=cr_count+1
             '   end if            
       '  response.write raw_p(jk) & " " & round(cr(zgroup,0),4) & "  " &  round(cr(zgroup,1),8) & " " & pattern_prob  & "<br>"         
         next
    'raw_p(jk)= sum percentage
 
  
  

  zzzz=0:zzzz2=0
 for jk=1 to krange
            zgroup=jk
       if group_p(zgroup,0)>0 and pattern_prob>0 then
          cr(zgroup,2)=   cr_count* cr(zgroup,1)/pattern_prob 'expected count
         ' response.write jk &"   " & round(cr(zgroup,2),4) &"<br>"
       end if
  next   

    'response.write "==================="  & zzzz & " bb" &  cr_count & "   " & zzzz2  & "<br>"

     chi_fit2=0: g_two2=0: CR32=0:cr_count=0
    for j =1 to  krange
          if group_p(j,0)>0 and cr(j,2)>0  then
               chi_fit2=chi_fit2 + cr(j,0)*((cr(j,0)/cr(j,2))-1) 
               g_two2= g_two2 +  cr(j,0)*((cr(j,0)/cr(j,2))^0.001-1) 
         
                CR32=CR32+  cr(j,0)*((cr(j,0)/cr(j,2))^0.67-1) 
               cr_count=cr_count+1
    
  
         end if
    next

   



tvalue=msumaad 
 df=krange-1
  strSQL =  "Select * From chiquest where a0=" & round(tvalue,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
              if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df   
                   pro=round(objrsst(itema),3)
                 end if
              end if
        
  dfdd="df" & krange-1
 
    if pro <0.05 then
         tvalue2= " <font color=red>" & round(pro,6) &"</font>"
             if pro=0 then
 tvalue2= " <font color=red>" & "<.0001" &"</font>"
              end if
   else
           tvalue2=round(pro,6) 
    end if
 


 
 
  redim exp_groupcr2(max_group)
  for jk=1 to mrow 
        
        exp_groupcr2(conabc2(jk))= exp_groupcr2(conabc2(jk))+person_catprob(jk)
 
  next  
 


 for j =0 to  max_group
    exp_groupcr2(j)= round(mrow* exp_groupcr2(j)/person_catprobabc,2)
next 
  'conabc2(jk) 'group number
  'mrow personno
  'person_catprobabc 'total proportion
   ' person_catprob(jk) proportion
 

  for j=1 to itemno
    redim exp_groupcr2(max_group)
     for jk=1 to mrow        
        exp_groupcr2(conabc2(jk))= exp_groupcr2(conabc2(jk))+test_prob(jk,j)
      next  

 next



 

tvalue=msumaad 
 df=max_group-1
  strSQL =  "Select * From chiquest where a0=" & round(tvalue,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
              if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df   
                   pro=round(objrsst(itema),3)
                 end if
              end if
         	
 
  dfdd="df" & krange-1
 
    if pro <0.05 then
         tvalue2= " <font color=red>" & round(pro,3) &"</font>"
   else
           tvalue2=round(pro,3) 
    end if




 
'===========================================

       chi_fit=2/(1+1)*chi_fit
      g_two= 2/(0.001*(0.001+1))*g_two 
      CR3= 2/(0.67*(0.67+1))*CR3
     chi_fit2=2/(1+1)*chi_fit2
      g_two2= 2/(0.001*(0.001+1))*g_two2 
      CR32= 2/(0.67*(0.67+1))*CR32
 
 zscore_mean=zscore_mean/(personno*itemno) 
 zscore_SD=(zscore_SD-(personno*itemno)*zscore_mean*zscore_mean)/(personno*itemno-1) 
   zscore_SD=zscore_SD^0.5
 
  perfect_i=perfect_i/personno :perfect_p=perfect_p/itemno
if request("covid")=""  then 
%>
   <table><tr><td>No.</td><td>Person</td><td>Grade</td><td>Theta</td><td>Model SE</td><td>InfitMNSQ</td><td>OutfitMNSQ</td><td>Chi_q</td><td>Raw Score</td><td>2 SE(infit)</td><td>2 SE(outfit)</td><td>InfitZSTD</td><td>OutfitZSTD</td><td>Corr.</td><td>Relia_I</td><td>Relia_O</td></tr>
<%
 end if
 ArrayIn= person
   ' For i = LBound(ArrayIn) To UBound(ArrayIn)
      '   For j = i + 1 To UBound(ArrayIn)
            ' If ArrayIn(i) > ArrayIn(j) Then
              '   SrtTemp = ArrayIn(j)
               '  ArrayIn(j) = ArrayIn(i)
              '   ArrayIn(i) = SrtTemp
               '   SrtTemp2 = raw_p(j)
              '   raw_p(j) = raw_p(i)
               '  raw_p(i) = SrtTemp2
              '   SrtTemp3 = var_p(j)
               '  var_p(j) = var_p(i)
               '  var_p(i) = SrtTemp3
                 
           '  End If
        ' Next  
    ' Next  
      allscoreuper=maxcat*itemno
      allscorelower=0
           firsta=0:allmean2=0
       redim outfitabc(personno), infitabc(personno)
       for jk=1 to personno
               allmean2=allmean2+person(jk)   
          if raw_p(jk)<>0 and firsta=0 then
              labela=raw_p(jk)
              labelb=arrayIn(jk)
              labelc=var_p(jk)
              firsta=1
         end if
             if  firsta=1 and raw_p(jk)>labela then
                    labela2=raw_p(jk)
                    labelb2=arrayIn(jk)
                    labelc2=var_p(jk)  'se
                 exit for
             end if
 
        next 
 
          if (labela2-labela)>0 then
                ratea=abs((labelb2-labelb)/(labela2-labela))
                zero_p=labelb-(abs(0-labela)+0.6)*ratea
                ratea_se=abs((labelc2-labelc)/(labela2-labela))
                zero_p_se=labelc+(abs(0-labela)+0.6)*ratea_se
         else
                 ratea=0
                zero_p=0
                ratea_se=0
                zero_p_se=0
         end if

          firsta=0
       for jk=personno to 1 step -1
 
         if raw_p(jk)<>itemno*1 and firsta=0 then
              labela=raw_p(jk)
 
              labelb=arrayIn(jk)

              labelc=var_p(jk)
              firsta=1
         end if

             if  firsta=1 and raw_p(jk)<labela then
                    labela2=raw_p(jk)
                    labelb2=arrayIn(jk)
                    labelc2=var_p(jk)  'se
                 exit for
             end if
        next
      
             if (labela2-labela)<>0 then 
                 ratea=abs((labelb2-labelb)/(labela2-labela))
                large_p=labelb+(abs(itemno*1-labela)+0.6)*ratea
                  ratea_se=abs((labelc2-labelc)/(labela2-labela))
                large_p_se=labelc+(abs(itemno*1-labela)+0.6)*ratea_se
            end if
                ' for perfect score above
          residualz=0:varz=0:jk=0:mean=0:sq=0:outfitran=0
       redim infitz(personno)


   if logtrans<4 then 
    for jk=1 to personno
                residualz=0:varz=0:outfitran=0
               for j=1 to itemno
                   logit=person(jk)-item(j)
                   exp_a=exp(logit)/(1+exp(logit)) 'prob.      
                   var_a = exp_a*(1-exp_a) 
                  var(jk,j)= var_a
                  
                  varz=varz+var(jk,j)
                  residualz=residualz+residual(jk,j)^2
 
               next
                 outfitran=outfitran+ residualz/itemno
                  infitz(jk)= round(residualz/varz  ,2) 
                 mean=mean+infitz(jk)
                 sq=sq+infitz(jk)^2
                  'sd=sd^.05
    'response.write round(1/varz^0.5,2) &"<br>"  
 
  
       next

             outfit_mean=round(outfitran/personno,2)
            mean=mean/personno
            sd=sq/personno-mean^2
    end if 'logtrans=3

      
      TSS=0    
   redim outfitz(personno)
 mmaxp=-100:mminp=100: mmaxinfit=0:mmininfit=100:mmaxoutfit=-110:mminoutfit=100
 redim searr(personno)
 redim measurearr(personno)
         dd1mean=0:dd1var=0:dd2mean=0:dd2var=0:dd3mean=0:dd3var=0:dd4mean=0:dd4var=0
         dd1max=0:dd1min=330:dd2max=0:dd2min=330:dd3max=0:dd3min=0:dd4max=0:dd4min=0
         adjse=0:modeladjse=0
        allmean2=0:  
    for jk=1  to personno      
                allmean2=allmean2+person(jk)   
            if person(jk)>10 then person(jk)=10 
            if person(jk)<-10 then person(jk)=-10
                   outfit=0:residualz=0:varz=0:rawtotal=0
                   infitCI=0:infitCI2=0:outfitCI=0 
                     PTMEASUR=0:item_se=0:item_se2=0
       
         '  zscore_sd= 0.1083 + -0.03087*(zscore_sd^2) +  2.0747*(zscore_sd^2)^2
    
         ' zscore_sd= zscore_sd^0.5



      for j=1 to itemno
       if isnumeric(test(jk,j))=true then  
if logtrans<4 then
            
                   if zscore_sd>0 then     
                     zscore(jk,j)=(zscore(jk,j)-zscore_mean)/zscore_sd ' зǤ outfit
                   else
                         zscore(jk,j)=0
                   end if
            
              if var(jk,j)=0 then var(jk,j)=0.001
 end if


                  outfit=outfit+zscore(jk,j)^2
    
                  varz=varz+var(jk,j)
 
                  residualz=residualz+residual(jk,j)^2
                  rawtotal=rawtotal+test(jk,j)
 
                    infitCI=infitCI + kurtosis(jk, j)
                    infitCI2=infitCI2 + var(jk, j)  
                   outfitCI=outfitCI + kurtosis2(jk, j)
                   PTMEASUR=PTMEASUR+(test(jk,j)-raw_p_avg(jk))*(raw_iz(j)-allscore9_p)
                    item_se=item_se+(test(jk,j)-raw_p_avg(jk))^2
                    item_se2=item_se2+(raw_iz(j)-allscore9_p)^2  
                       TSS=TSS+(test2(jk,j)-allmean)^2
                 '  response.write round(tss,2) & "    " & test2(jk,j) & "      " & round(allmean,2) & "     " & round((test2(jk,j)-allmean)^2,2) &  "<br>"
           end if
  
     next 'for item


        if item_se=0 or item_se2=0 then
                  PTMEASUR=0
               elseif item_se>0 and item_se2>0 then
                 PTMEASUR= PTMEASUR/(item_se*item_se2)^0.5 
                else
                  PTMEASUR=0 
                end if
                score8 =round(PTMEASUR,2) 
               
                'objrs2("infit")=round(residualz/varz,2) 
                  
         if item_se>0 and outfit>0 then
          outfit=round(outfit/itemno,2)
         else
            outfit=1
         end if
        outfitabc(jk)=outfit
        if outfitabc(jk)=0 then outfitabc(jk)=1

     
        if sd>0 and varz>0 then
            if mean<1 then
                  infit=round((residualz/varz+(1-mean)),2) 
            else
                infit=round((residualz/varz-(mean-1)),2)
            end if 

        elseif varz>0 then
                if mean<1 then
                    infit=round((residualz/varz+(1-mean)),2) 
                 else
                   infit=round((residualz/varz-(mean-1)),2)
                 end if 
        else
             infit=1
         end if
       infitz(jk)= infit
       if infit>10 then infit=9.9
       if infit<0 then infit=0.01
        
   if logtrans<>3 and varz>0 then
     infit=round(residualz/varz,2)
     infitz(jk)= round(residualz/varz  ,2) 
   end if
     if infitz(jk)=0 then infitz(jk)=1
          


 'response.write residualz&":" & varz & ":aaa" & mean &":ddd" & round(sd,5) &"<br>"
  'response.write infit &":" & sd & ":" & varz &":" & mean
  
        if infitCI>0 and outfitCI>0 and InfitCI2>0 then
                   
              InfitCIse=round((InfitCI/InfitCI2^2/(itemno))^0.5,2) 
  if (itemno-perfect_i)>0 and (itemno-perfect_i)>0 and (itemno-perfect_i)>0 then
   maa=((outfitCI/((itemno-perfect_i)^2)-1/(itemno-perfect_i))/(itemno-perfect_i))
  else
   maa=0
  end if
                  if maa>0 then 
                    outfitCIse=round(maa^0.5,2)
                  else
                     outfitCIse=0.001
                   end if 

             mmark=""
             if (personno-perfect_p)>0 then
               if  outfit -2*outfitCIse/(personno-perfect_p)^.5 >2   then
                    ' mmark="<font color=red>*</font>"
               end if
             end if

            if outfit>=2 then
                mmark="<font color=red>*</font>"
            end if



       if varz>0 then


         if round((residualz/varz+(1-mean)),2)=0 or round(outfit/itemno,2)=0  or InfitCIse=0 or OutfitCIse=0 then
           infitzstd=0:outfitzstd=0
         else

             if InfitCIse>0 and OutfitCIse>0 and itemno-perfect_i>0 then 
 


            infitzstd=(infit^0.33-1)*(3/(InfitCIse^2*(itemno))^0.5)+((InfitCIse^2*(itemno))^0.5/3)
 
            outfitzstd=(outfit^0.33-1)*(3/(OutfitCIse^2*(itemno-perfect_i))^0.5)+((OutfitCIse^2*(itemno-perfect_i))^0.5/3)  
             else
                 infitzstd=0:outfitzstd=0
             end if
         end if
       end if


       else
           infitzstd=0:outfitzstd=0
        end if
  se  =round(1/varz^0.5,2) 
 
 
 if logtrans>3 then 
 
           if extremeshiftp=1 and vextremeperson>0 and vextremeperson0>0 then  
            if raw_p(jk)=maxcat*itemno then
                   varz= vextremeperson 
                 var_p(jk)=varz
            end if
          
              if raw_p(jk)=0  and vextremeperson0>0 then 
                varz=round(1/sqr(vextremeperson0),2) 
                   var_p(jk)=varz
              else
                
              end if
    
            end if ' if extremeshiftp=1

                 varz=var_p(jk)
                  se  =round(1/varz^0.5,2)
 end if '>3 


              if se>1.84 then se=1.84
            score4 =round(zscore_mean,2)
              score5 =round(zscore_sd,2)  ' prepare used in Kidmap
              score6 =round(infitzstd,2) 
                score7 =round(outfitzstd,2)   
     
           score9= raw_p(jk)
outfitz(jk)=outfit
 infitz(jk)=infit
searr(jk)=se

measurearr(jk)= person(jk)

relia_p=   ((1-exp(outfit)/(1+exp(outfit)))*2)^0.48 
  relia_i=   ((1-exp(infit)/(1+exp(infit)))*2)^0.48 
 personlabel="" & jk
  
 if   personnamezz>"" and ubound(personname2)>=jk-1 then

     personlabel=personname2(jk-1)
 end if
  if infit=0 then infit=1
  if outfit=0 then outfit=1

                           kidat=round((6-person(jk))*2,0)
               if kidat<0 then kidat=0
               if kidat>25 then kidat=25
                 ranking=Fix(kidat/5)+1
                 alpheta="ABCDEFGJIJK"
                 remainer=kidat mod 5 
       
              groupa= Cdbl(conabc2(jk))
               rk=mid(alpheta,ranking,1) & right("0" & (remainer+1),2) 
     


 
          if extremeshiftp=1 and vextremeperson>0 and vextremeperson0>0 then  
                if raw_p(jk)=maxcat*itemno then
                   varz= vextremeperson                
                  if varz>1/(1.84*1.84) then varz=1/(1.84*1.84) 
                  if varz=0 then varz=0.001                   
                     var_p(jk)=varz 
                     se=1/sqr(varz)                               
                 end if 
                 if raw_p(jk)=0 then  
                   varz=vextremeperson0
                     if varz>1/1.84*1/1.84 then  varz=1/1.84*1/1.84
                     if varz=0 then varz=0.001
                       var_p(jk)=varz
                     se=1/sqr(varz)
                 end if
           end if 


     if se>1.84 then se=1.84

if request("covid")="" then 
  %>
<tr><td> <%=personlabel%></td><td><%=person_gp(jk)%> </td><td><%=personrk(jk)%></td><td><%=round(person(jk),2)%></td><td><%=se%></td><td><%=infit%></td><td><%=outfit%><%=mmark%></td><td><%=chisuareabc2(jk)%></td><td><%= round(raw_p2(jk),2)%></td><td><%=2*infitcise%></td><td><%=2*outfitcise%></td><td><%=round(infitzstd,2)%></td><td><%=round(outfitzstd,2)%></td><td><%=score8%></td><td><%=round(relia_i,2)%><td><%=round(relia_p,2)%> </td></tr>
 <% end if 


        if infit>1 then
            adjse=adjse+infit *se*se
        else
               adjse=adjse+se*se
        end if 
         modeladjse=modeladjse+se*se

        dd1var=dd1var+outfit^2
       dd1mean=dd1mean+outfit
         if  outfit>dd1max then dd1max=outfit
        if  outfit<dd1min then dd1min=outfit
       
        dd2var=dd2var+infit^2
       dd2mean=dd2mean+infit
         if infit>dd2max then dd2max=infit
        if  infit<dd2min then dd2min=infit
    
     
        dd3var=dd3var+ outfitzstd^2
       dd3mean=dd3mean+ outfitzstd
         if   outfitzstd>dd3max then dd3max= outfitzstd
        if   outfitzstd<dd3min then dd3min= outfitzstd
       
        dd4var=dd4var+infitzstd^2
       dd4mean=dd4mean+infitzstd
         if infitzstd>dd4max then dd4max=infitzstd
        if  infitzstd<dd4min then dd4min=infitzstd
    
    if person(jk)> MMAXP then mmaxp=person(jk)
    if person(jk)<MMINP then mminp=person(jk)

    if infitz(jk)> mmaxinfit then mmaxinfit=infitz(jk)
     if infitz(jk)< mmininfit then mmininfit=infitz(jk)


     if outfitz(jk)> mmaxoutfit then mmaxoutfit=outfitz(jk)

     if outfitz(jk)< mminoutfit then mminoutfit=outfitz(jk)

     next 'jk
 
     modeladjse=round(sqr(modeladjse/personno),2)
     adjse=sqr(adjse/personno)
     ' if adjse<1 then adjse=.4
       adjse=round(adjse,2)
  dd1mean=round(dd1mean/personno,2) 
   dd1var=round(sqr((dd1var/personno-dd1mean*dd1mean)),2)  
  dd2mean=round(dd2mean/personno,2) 
   if dd2var/personno-dd2mean*dd2mean>0 then 
   dd2var=round(sqr((dd2var/personno-dd2mean*dd2mean)),2)
   else
       dd2var=0.001
   end if  
  dd3mean=round(dd3mean/personno,2) 

   dd3var=round(sqr((dd3var/personno-dd3mean*dd3mean)),2)  
  dd4mean=round(dd4mean/personno,2) 
   dd4var=round(sqr((dd4var/personno-dd4mean*dd4mean)),2)  
'response.write "AAAAAAAAAAAAAA"

 'response.write dd1mean & " " &  dd2mean & " " &  dd3mean& " " &  dd4mean &"<br>"
 ' response.write dd1var & " " &  dd2var & " " &  dd3var& " " &  dd4var&"<br>"
  'response.end

'=========person  theta==================================
 if max_group>=1 then
  
 
                        TSSb=0:  TSS=0
                          redim g_mean(max_group)
                             redim Group_num(max_group)
                   

                                 allmean2=allmean2/personno
                  
                for jk=1 to  personno

 
                     g_mean(conabc2(jk))= g_mean(conabc2(jk))+ person(jk) 'raw_p2(jk)
                                   Group_num(conabc2(jk))=Group_num(conabc2(jk))+1                                     
                                     TSS=TSS+(person(jk)-allmean2)^2

                next  
                      
                for j=0 to  max_group
                   if Group_num(j)>0 then
                     g_mean(j)= g_mean(j)/Group_num(j)
                 else
                     g_mean(j)=0
                 end if
                  
                next  

                       for jm=0 to max_group
                             TSSb=TSSb+ Group_num(jm)* (allmean2-g_mean(jm))^2
                      ' response.write round(allmean2*itemno,2) & " dd" & g_mean(jm) &  "ddddd<br>"
                       next


                   'ANOVA ================
           
                        dft=personno-1
                        dfb=max_group+1-1
                        dfw=dft-dfb  
           
             
                                 dfw=0
                              for j=0 to max_group
                                dfw= dfw+ Group_num(j)-1
  
                              next 
 
                        TSSw=TSS-TSSb
                       if dfb>0 and dft-dfb>0 then
                        MSb=TSSb/dfb
                        MSw=TSSw/(dft-dfb)
                       else
                         MSb=0
                        MSw=0
                      end if
                    if msw>0 then
                        Fstatistic=round(msb/msw,2)
                     else
                         Fstatistic=1
                     end if
                        p_F="=FDIST(" & Fstatistic &"," & dfb &"," & dfw &")"
                    '====================

 
 %>
</table>
<table style="width:100%"><tr><td><font color=red>ANOVA</font></td><td>Virable</td><td>SS</td><td>df</td><td>MSS</td><td>F</td><td><font color=red>p</font></td></tr>
        <tr><td> </td><td>Between</td><td><%=round(TSSb,2)%></td><td><%=dfb%></td><td><%=round(MSb,2)%></td><td><%=round(Fstatistic,2)%></td><td><%=p_F%></td></tr>
        <tr><td> </td><td>Within</td><td><%=round(TSSw,2)%></td><td><%=dfw%></td><td><%=round(MSw,2)%></td><td></td><td><a href="https://www.socscistatistics.com/pvalues/fdistribution.aspx" target=blank><font color=red>p-value(Click on Me)</font></a></td></tr>
        <tr><td> </td><td>TSS</td><td><%=round(TSS,2)%></td><%=personno*itemno-1%><td><%=dfw+dfb%> </td><td> </td><td><font color=red>All mean= </font></td><td><%=round(allmean,2)%> </td></tr>
</table>
<br><br><br>
<table> 
 <tr><td>Group</td><td>Count</td><td>Total</td><td>Mean</td><td>Varince</td><td>Tlogit</td><td>Logt</td><td>Var</td></tr>
<%
 
    
 
     for j=min_group to max_group-1
         avgb=0: sumsqb=0: avgb2=0: sumsqb2=0: nnn=0
       for jk=1 to personno
          if j=cint(conabc2(jk)) then
              avgb=avgb+ raw_p2(jk)
              sumsqb=sumsqb+raw_p2(jk)*raw_p2(jk)
              nnn=nnn+1
                 avgb2=avgb2+ person(jk)
               sumsqb2=sumsqb2+person(jk)*person(jk)
           end if
         next 
        if nnn=0 then nnn=1
              tavgb=avgb
              avgb=round(avgb/nnn,2)

             varb=round(sumsqb/nnn-avgb*avgb,2)
           tavgb2=round(avgb2,2)

              avgb2=round(avgb2/nnn,2)
             varb2=round(sumsqb2/nnn-avgb2*avgb2,2)
  ' response.write "<tr><td>" & j & "</td><td>" & nnn  & "</td><td>" & tavgb  & "</td><td>" &  avgb& "</td><td>" &   varb&"</td><td>" & tavgb2  & "</td><td>" &  avgb2& "</td><td>" &   varb2&"</td></tr>"
       
     next
 
 ' response.write "</table>"
 'response.write "<br>"   
'response.write "<br>"  
'response.write "<br>"  
      if request("covid")="12" then
           response.end
 end if  
     
%>
<table style="width:100%"><tr><td>person</td><td>Outfit</td>
<%
     for j=1 to itemno
      response.write "<td>" & j & "</td>"
     next

%>
</tr><tr><%
 for jk=1 to personno
 personlabel=  jk & " " & personname2(jk)

' if personnamezz>"" and ubound(personname2)>=jk-1 then
    ' personlabel=personname2(jk-1)
' end if
    response.write "<td>" & personlabel &"</td></td><td>" & round(outfitz(jk) ,2) & "</td>" 
                for j=1 to itemno
                 if isnumeric(test(jk,j))=true then
                       ' zscore(jk,j)=(zscore(jk,j)-zscore_mean)/zscore_sd
                  if round(zscore(jk,j),2)>=2 then
                       response.write "<td><img src='../kpiall/redlight.png' alt='Under Fit' height='12' width='12'></td>"

                  elseif round(zscore(jk,j),2)<=-2 then
                       response.write "<td><img src='../kpiall/bluelight.png' alt='Over Fit' height='12' width='12'></td>"
                  else
                    response.write "<td> </td>"
                  end if
                 else
                     response.write "<td> </td>"
                  end if

                next
    response.write "<tr>"
  next
  
%>
</table>
<table style="width:100%"><tr><td><font color=red>Z-score</font></td><td></td></table>
<table style="width:100%"><tr><td>Person</td><td>Outfit2</td>
<%
     for j=1 to itemno
      response.write "<td>" & j & "</td>"
     next
%>
</tr><tr><%
 for jk=1 to personno
  personlabel="A" & jk
 if personnamezz>"" and ubound(personname2)>=jk-1then
     personlabel=personname2(jk-1)
 end if
   response.write "<td>" & personlabel &"</td><td>" & round(outfitz(jk),2) & "</td>" 
                for j=1 to itemno
                     if isnumeric(test(jk,j))=true then
                       ' zscore(jk,j)=(zscore(jk,j)-zscore_mean)/zscore_sd
                       za = round(zscore(jk,j),2)
                  if za>=2 then
                       response.write "<td><font color='#ff0000'>" & za  &"</font></td>"

                  elseif za<=-2 then
                       response.write "<td><font color='#00ff00'>" & za  &"</font></td>"

                  else
                      response.write "<td>" & za  &"</td>"
                  end if
                   else
                      response.write "<td>" & za  &"</td>"
                  end if

                next
    response.write "<tr>"
 next
         
%>
</table>
<% if logtrans<4 then %>
<table style="width:100%"><tr><td><font color=red>expectation</font></td><td></td></table>
<table style="width:100%"><tr><td>Person</td><td>Outfit2</td>
<%
     for j=1 to itemno
      response.write "<td>" & j & "</td>"
     next
%>
</tr><tr><%
 for jk=1 to personno
  personlabel="A" & jk
 if personnamezz>"" and ubound(personname2)>=jk-1then
     personlabel=personname2(jk-1)
 end if
   response.write "<td>" & personlabel &"</td><td>" & round(outfitz(jk),2) & "</td>" 
                for j=1 to itemno
                  if isnumeric(test(jk,j))=true then
                       prob=(person(jk)-item(j))
                       prob=exp(prob)/(1+exp(prob))
                       za = round(prob,2)
                        
                      response.write "<td>" & za  &"</td>"
                   
                   else
                     za="."
                      response.write "<td>" & za  &"</td>"
                  end if

                next
    response.write "<tr>"
 next
         
%>
</table>

<table style="width:100%"><tr><td><font color=red>percentage</font></td><td></td></table>
<table style="width:100%"><tr><td>Person</td><td>Outfit2</td>
<%
     for j=1 to itemno
      response.write "<td>" & j & "</td>"
     next
%>
</tr><tr><%
 for jk=1 to personno
  personlabel="A" & jk
 if personnamezz>"" and ubound(personname2)>=jk-1then
     personlabel=personname2(jk-1)
 end if
   response.write "<td>" & personlabel &"</td><td>" & round(outfitz(jk),2) & "</td>" 
                for j=1 to itemno
                  if isnumeric(test(jk,j))=true then
                       prob=(person(jk)-item(j))
                       prob=exp(prob)/(1+exp(prob))
                       za = round(prob,2)
                        za = test(jk,j)

                      response.write "<td>" & za  &"</td>"
                   
                   else
                     za="."
                      response.write "<td>" & za  &"</td>"
                  end if

                next
    response.write "<tr>"
 next
         
%>
</table>
<% end if %>
<% jk=1

 j2=0 : score9_var2=0
   for j=1 to itemno
       score9_var2=score9_var2+ raw_iz(j)^2
        j2=j2+raw_iz(j)          'objrs2("score9") ' `         
   next
 
  j2=j2/itemno
  score9_var2=(score9_var2-(itemno)*j2*j2)/(itemno-1) 
 sumraw_p2=0  
for j=1 to personno
    jk2=0 : score9_var=0
    for jk=1 to itemno
        score9_var=score9_var+test2(j,jk)^2 '% score
        jk2=jk2+test2(j,jk) ' `       
    next  
   sumraw_p2= sumraw_p2+ raw_p2(j)  
   jk2=round(jk2/itemno,2)
    var_item =var_item + (score9_var-(itemno)*jk2*jk2)/itemno
cronbanalpha2=0
if score9_var2>0 then
    cronbanalpha2=round(itemno/(itemno-1)*(1- var_item/score9_var2),2)
end if
 ' response.write  j & " " & jk2   & " " & personno  & " " & round(score9_var,2)  & " " & round((score9_var-(personno)*jk2*jk2)/(personno-1),2) & "<br>"  
next
meanraw_p2=sumraw_p2/personno
 


 if cronbanalpha2>1 then cronbanalpha2=1



'===XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX====================================
   jk2=0 : score9_var=0:rawmax=0:rawmin=20000
   measuremean=0 : measurevar=0:semean=0: semean2=0: sevar=0
   semax=-10: semin=2222: mcount=0:meanrawp2=0
   for jk=1 to personno
     ' if 1/var_p(jk)>1.84*1.84 then var_p(jk)=(1/1.84)*(1/1.84)
     ' if person(jk)>6.07 then person(jk)=6.07
       score9_var=score9_var+raw_p2(jk)^2
             'objrs2("score9") ' `   
        if  raw_p2(jk)>rawmax then rawmax=raw_p2(jk)
        if  raw_p2(jk)<rawmin then rawmin=raw_p2(jk) 

           if extremeshiftp=1 and vextremeperson>0 and vextremeperson0>0 then  
                if raw_p(jk)=maxcat*itemno then
                   varz= vextremeperson                
                  if varz>1/(1.84*1.84) then varz=1/(1.84*1.84)                    
                     var_p(jk)=varz                                
                 end if 
                 if raw_p(jk)=0 then  
                   varz=vextremeperson0
                     if varz>1/1.84*1/1.84 then  varz=1/1.84*1/1.84
                       var_p(jk)=varz
                 end if
           end if 
       
       
      if raw_p(jk)=maxcat*itemno or raw_p(jk)=0 then

      else
        meanrawp2 = meanrawp2 + (raw_p2(jk)-meanraw_p2)^2
        measurevar=measurevar+person(jk)^2 
        masuremean=measuremean+ person(jk) 
           jk2=jk2+ raw_p2(jk)  
        sevar=sevar+1/var_p(jk)
        semean=semean+1/(var_p(jk)^.5) 
        semean2=semean2+1/(var_p(jk)) 
         mcount= mcount+1
      end if
         if  1/var_p(jk)^0.5>semax then semax=1/var_p(jk)^0.5
        if  1/var_p(jk)^0.5<semin then semin=1/var_p(jk)^0.5  
    ' response.write round(person(jk),2) &"," & round(1/var_p(jk),2) & "," &  semean2& "<br>"
      'response.write jk &" " & 1/var_p(jk)^.5   &"<br>"

   next  
   meanrawp2=meanrawp2/(mcount-1)

  jk2=jk2/mcount
  score9_var2=(score9_var/mcount-k2*jk2) 
  rawmean=round(jk2,1)
measuremean=round(measuremean/mcount,2)
 
measurevar=round(measurevar/mcount-measuremean*measuremean,2) 
 semean=round(semean/mcount,2)
   
 semean2=round(semean2/mcount,2) 
  
 sevar=(sevar/mcount-semean*semean)
 if sevar<0 then sevar=.11
if measurevar-semean2>0 then
   sap=round(measurevar-semean2,2)
else
   sap=round(measurevar-measurevar/4,2)
end if
  modelsap2=sap



 if measurevar-adjse*adjse>0 then
   adjsap=round(measurevar-adjse*adjse,2)
   modeladjsap=round(measurevar-modeladjse*modeladjse,2)
 else
  adjsap=round(measurevar-measurevar/4,2)
  modeladjsap=adjsap
 end if
 'response.write semean2 & "<br>"
 'response.write semean  & "<br>"
 'response.write adjse*adjse & "<br>"
   if sap<=0 then sap=.001
if semean2<=0 then semean2=.001
 
 spmodelG=round((sap/semean2)^.5,2) 
   if adjsap<=0 then adjsap=.001

 sprealG=round((adjsap/(adjse*adjse))^.5,2)
 if spmodelG<=0 then spmodelG=.001
 reliamodel=round(spmodelG^2/(1+spmodelG^2),2)
  reliareal=round(sprealG^2/(1+sprealG^2),2)
'++++++++++++++++++++++++++++++++
 spmodelG2=round((modeladjsap/(modeladjsap*modeladjsap))^.5,2)
  spmodelG2=round(sap/modeladjse,2) 
 if spmodelG2<=0 then spmodelG2=.001
 reliamodel2=round(spmodelG2^2/(1+spmodelG2^2),2)
 



semeani=0: semeani2=0: sevari=0:semaxi=0:semini=333
meameani=0:measdi=0:meamaxi=0:meamini=300
 var_item=0:rawmeani=0:rawsdi=0:rawmaxi=0:rawmini=300
  mcount2=0
  pqvar=0
for j=1 to itemno
    jk2=0 : score9_var=0
     pcount=0:qcount=0
    for jk=1 to personno
        score9_var=score9_var+test2(jk,j)^2 '% score
        jk2=jk2+test2(jk,j) '
         if test2(jk,j) =1 then pcount=pcount+1
         if test2(jk,j) =0 then qcount=qcount+1
     next
      pqvar=pqvar+pcount/personno*qcount/personno

   jk2=round(jk2/personno,2)
    var_item =var_item + (score9_var-(personno)*jk2*jk2)/(personno)
    rawmeani=rawmeani+raw_i(j)
    rawsdi=rawsdi+raw_i(j)*raw_i(j)
        if  raw_i(j)>rawmaxi then rawmaxi=raw_i(j)
        if  raw_i(j)<rawmini then rawmini=raw_i(j) 

  
        if  item(j)>meamaxi then meamaxi=item(j)
        if  item(j)<meamini then meamini=item(j) 



      if raw_i(j)=maxcat*personno or raw_i(j)=0 then

      else
            meameani=meameani+item(j)
            measdi=measdi+item(j)*item(j)  
         semeani=semeani+1/(item_var(j)^.5) 
         sevari= sevari+1/(item_var(j)) 
         semeani2=semeani2 +1/(item_var(j)) 
         mcount2= mcount2+1

      end if
         if  1/item_var(j)^0.5>semaxi then semaxi=1/item_var(j)^0.5
        if  1/item_var(j)^0.5<semini then semini=1/item_var(j)^0.5 
   ' response.write  j & " " & jk2   & " " & personno  & " " & round(score9_var,2)  & " " & round((score9_var-(personno)*jk2*jk2)/(personno-1),2) & "<br>"  
   'response.write    (score9_var-(personno)*jk2*jk2)/(personno) & "<br>"
next
   rawmeani= round(rawmeani/mcount2,2)
    rawsdi=round(rawsdi/mcount2-rawmeani*rawmeani,2)
   meameani= round(meameani/mcount2,2)
    measdi=round(measdi/mcount2-meameani*meameani,2)
      semeani=round(semeani/mcount2,2)
     semeani2=round(semeani2/mcount2,2)
 sevari=(sevari/mcount2-semeani*semeani)
 if sevari<0 then sevari=.11

 if score9_var2=0 then score9_var2=0.01
 cronbanalpha=0
if score9_var2>0 then
  cronbanalpha=round(itemno/(itemno-1)*(1- var_item/score9_var2),2)
end if



if cronbanalpha>1 then cronbanalpha=1
  'response.write  itemno & " " & round(var_item,2) & " " & round(score9_var2,2)

           tvalue=chi_kendall       
       df=dfk-1
          if  df >100 then
                df=100
             else
                  if tvalue>49 then
                     tvalue=round(tvalue/(tvalue/50),2)
                      df=round(df/(tvalue/50),2)
                  end if
      
  strSQL =  "Select * From chiquest where a0=" & round(tvalue,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
              if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df   
                   pro=round(objrsst(itema),3)
                 end if
              end if
           end if



if request("covid")=""  then 
%>
<table style="width:100%"><tr>corr_k is the average (Spearman) correlation coefficient computed on the ranks of all pairs of raters.</tr></table>
 <table><tr>http://www.real-statistics.com/reliability/kendalls-w/</tr></table>
<table><tr><td>Kendall_w</td><td>df</td><td>chisquare(Col=item ,row for judge)</td><td>corr_k for W</td><td>Cronban alpha (Col=person)</td><td>Cronban alpha(Col=item, in tradition) </td></tr>
 <tr><td><%=kendall_w%></td><td><%=dfk-1%></td><td>=round(pro,2)</td><td><%= corr_k%></td><td><%=cronbanalpha2%> </td><td><%=cronbanalpha%> </td></tr> 

</table>
 <% end if
    
    if request("covid")="10" or request("covid")="11" then
          For jkm = 1 To category_number - 1 ' mincat
                catadj(jkm)=catthresh(jkm)-cat_avg
              catcalibrate(jkm)=catadj(jkm)
         response.write "Steps..."  & JKM  & " " & round(catcalibrate(jkm),2) &"<br>"
          next
        response.write "iteration...<br>" &  iteration & "<br>"
     response.end
   end if

 
 
 if request("covid")="09" or request("covid")="29" then

       threshold=category_number-1
     response.write "<font color=red>Simulation data genetated(Copy & paste data below to MS Excel or other text file..</font>" & "<br>"
     mtext=""
      

  contentbb=split(content,chr(13))
   contentbb2=replace(contentbb(0),"	",",")
   contentbb3=split(contentbb2,",")
   for j=0 to ubound(contentbb3) 
        if j=0 then
            mtext=contentbb3(j)  
         else         
          mtext=mtext &"," & contentbb3(j)  
         end if         
    next 

                 if request("covid")="29" then
                      deltaabc=split(request("delta1"),",")
                      deltaabc2=split(request("delta2"),",") 
                            itemno=ubound(deltaabc)+1
                               for j=1 to itemno
                                  if j=1 then
                                     mtext="Item1"  
                                   else         
                                      mtext=mtext & ", Item" & j
                                   end if
                                next 
                                       mtext=  mtext &", name, group"
                         redim item(itemno)

                        for jjj=1 to itemno
                          item(jjj)=cdbl(deltaabc(jjj-1))
                        next

                        Category=ubound(deltaabc2)+2
                           redim catcalibrate(ubound(deltaabc2)+1)
                      
                         for jjj=1 to ubound(deltaabc2)+1
 
                           catcalibrate(jjj)=cdbl(deltaabc2(jjj-1))
                        next

                           threshold=ubound(deltaabc2)+1                        
               end if 
      response.write mtext  & chr(13) & "<br>" 

     cguess_a=0:apha_a=1
 for jk =1 to personno
   ability2=person(jk)
        contentbb2=replace(contentbb(jk),"	",",")
 
         contentbb3=split(contentbb2,",")

                      if request("covid")="29" then

                        if isnumeric(contentbb3(ubound(contentbb3)-1)) then
                          ability2=cdbl(contentbb3(ubound(contentbb3)-1))

                        end if
                      end if
 
        mscore=0
        scoretxt=""
 
    for j= 1 to itemno
            item_diff=item(j) 
 
             ReDim cumexp(threshold + 1)
          M = threshold + 1
 
          measure = 0
          cumexp(1) = 1
  For Category = 2 To M
    measure = measure + ability2 - item_diff - catcalibrate(Category - 1)
    FroExp = (1 - cguess_a) * Exp(apha_a * measure)
 
     cumexp(Category) = cumexp(Category - 1) + FroExp
  Next  
    Randomize
     mrnd = Rnd
        if cdbl(logtrans)=4 then   
             u = mrnd * cumexp(M)
                For Category = 1 To M
                  If u <= cumexp(Category) Then x = Category - 1: Exit For
                Next 
          else
         if  abs(zscore(jk,j))>2 then
           x=round(EXPECT(jk,j)*(kmax-kmin),2)
         else
              x=test2(jk,j)
         end if
          end if
         
  if logtrans<4 then
              logit1=person(personno)-item(1)
          prob1=exp(logit1)/(1+exp(logit1))
           x1=round(simulatemincat+(simulaterange)*prob1,0)
            minsimulatemincat=simulatemincat-x1
          
          logita=person(jk)-item(j)
       prob=exp(logita)/(1+exp(logita))

 
     x=round(simulatemincat+minsimulatemincat+(simulaterange)*prob,0)
    
          'test(jk,j)=round((test(jk,j)-mincat)/(maxcat-mincat),2)
             if j=1 then
                scoretxt=x
             else         
               scoretxt=scoretxt &"," & x  
             end if
  else
            if j=1 then
                scoretxt=x
             else         
               scoretxt=scoretxt &"," & x  
             end if

  end if
     
    next    
     response.write scoretxt &"," & contentbb3(ubound(contentbb3)-1)   &"," & contentbb3(ubound(contentbb3))   & chr(13) & "<br>"  
   
 next 
        response.write "<BR>"
 response.write "<BR>"
 response.write "<BR>"
 response.write "<BR>"
   response.end
  end if '09
 

%>
<table style="width:100%"><tr><td>Person</td><td>Infit</td>
<%
     for j=1 to itemno
      response.write "<td>" & j & "</td>"
     next
%>
</tr><tr><%
 for jk=1 to personno
  personlabel="A" & jk
 if personnamezz>"" and ubound(personname2)>=jk-1then
     personlabel=personname2(jk-1)
 end if
   response.write "<td>" & personlabel & "</td><td>" & round(infitz(jk) ,2) & "</td>" 
        for j=1 to itemno
              itemx="item" & j  
              if isnumeric(test(jk,j))=true then
                       ' zscore(jk,j)=(zscore(jk,j)-zscore_mean)/zscore_sd
                       za = round(zscore(jk,j),2)
                  if za>=2 then
                       response.write "<td><font color='#ff0000'>" & test2(jk,j) & "</font></td>"

                  elseif za<=-2 then
                       response.write "<td><font color='#00ff00'>" & test2(jk,j)  & "</font></td>"

                  else
                      response.write "<td>" & test2(jk,j) &"</td>"
                  end if
             else
                 response.write "<td>" & test2(jk,j) &"</td>"
              end if
           
                next
    response.write "<tr>"
  next
        
%>
</table>
<%
    ' for j=1 to itemno
     '  response.write "<td>" & j & "</td>"
     '   response.write "<td>" & item(j) & "</td><br>"
    ' next
  end if 'max-group>1


if request("covid")=""  then  
%>

<table style="width:100%">
</tr><td>
<input type="button" value="if Crashed then Refresh" onClick="location.href='raschrsm.asp?repno=<%=repno%>&myear=<%=myear%>'">
</td>
</tr>
</table>
 <table><tr><td>No.</td><td>Item</td><td>Difficulty</td><td>Model SE</td><td>ParaA</td><td>InfitMNSQ</td><td>OutfitMNSQ</td><td>Raw Score</td><td>2 SE(infit)</td><td>2 SE(outfit)</td><td>InfitZSTD</td><td>OutfitZSTD</td><td>Corr.</td><td>relia_I</td><td>relia_O</td></tr>
<% end if
ArrayIn= item
    redim itemserial(UBound(ArrayIn)+1)
        redim itemname2(UBound(ArrayIn)+1)
     For i = LBound(ArrayIn) To UBound(ArrayIn)-1 
           itemserial(i+1)=i+1
            itemname2(i+1)=itemname(i+1)
                             '   response.write itemserial(i+1) & 'itemname2(itemserial(i+1)) & round(ArrayIn(itemserial(i+1)),2)  & "<br>"
       next
     
      
    For i = LBound(ArrayIn) To UBound(ArrayIn)-1
         For j = i + 1 To UBound(ArrayIn)
             If cdbl(ArrayIn(i+1)) < cdbl(ArrayIn(j)) Then
                 SrtTemp = ArrayIn(j)
                 ArrayIn(j) = ArrayIn(i+1)
                 ArrayIn(i+1) = SrtTemp
                SrtTemp2 = raw_i(j)
                 raw_i(j) = raw_i(i+1)
                 raw_i(i+1) = SrtTemp2
               SrtTemp4 = item_var(j)
                 item_var(j) = item_var(i+1)
                 item_var(i+1) = SrtTemp4
                SrtTemp5 =itemserial(j)
                  itemserial(j) = itemserial(i+1)
                itemserial(i+1) = SrtTemp5
               SrtTemp6= itemname2(j)
             itemname2(j) = itemname2(i+1)
                 itemname2(i+1) = SrtTemp6
             End If
         Next  
     Next
    '  For i = LBound(ArrayIn) To UBound(ArrayIn)-1 
              '    response.write itemserial(i+1) & itemname2(itemserial(i+1)) & round(ArrayIn(itemserial(i+1)),2)  & "<br>"
        'next
      ' response.end   
         
    For i = LBound(ArrayIn) To UBound(ArrayIn)-1
          
            itemname2(i+1)=itemname(i+1)
                'response.write i+1 & itemname2(i+1) & "<br>"
       next
         
      allscoreuper=maxcat*personno
      allscorelower=0
           firsta=0
       for j=1 to itemno
              
         if raw_i(j)<>0 and firsta=0 then
              labela=raw_i(j)
              labelb=arrayIn(j)
              labelc=item_var(j)
              firsta=1
         end if
             if  firsta=1 and raw_i(j)>labela then
                    labela2=raw_i(j)
                    labelb2=arrayIn(j)
                    labelc2=item_var(j)  'se
                 exit for
             end if
        next 
 
          if (labela2-labela)<>0 then 
                ratea=abs((labelb2-labelb)/(labela2-labela))
                zero_item=labelb-(abs(0-labela)+0.6)*ratea
                ratea_se=abs((labelc2-labelc)/(labela2-labela))
                zero_item_se=labelc+(abs(0-labela)+0.6)*ratea_se
         end if
          firsta=0
       for j=itemno to 1 step -1
         if raw_i(j)<>personno*1 and firsta=0 then
              labela=raw_i(j)
              labelb=arrayIn(j)
              labelc=item_var(j)
              firsta=1
 
         end if
             if  firsta=1 and raw_i(j)<labela then
                    labela2=raw_i(j)
                    labelb2=arrayIn(j)
                    labelc2=item_var(j)  'se
                 exit for
             end if
 
        next 
 
              if (labela2-labela)>0 then
                 ratea=abs((labelb2-lableb)/(labela2-labela))
                large_item=labelb+(abs(personno*1-labela)+0.6)*ratea
                  ratea_se=abs((labelc2-lablec)/(labela2-labela))
                large_item_se=labelc+(abs(personno*1-labela)+0.6)*ratea_se
 
              end if
                ' for perfect score above
              ' response.write item(j) & " " & limit & " " & large_item & " " & zero_item
        

      
 response.write "==========================================================<br>"  
 

 
     dd1meani=0:dd1vari=0:dd2meani=0:dd2vari=0:dd3meani=0:dd3vari=0:dd4meani=0:dd4vari=0
         dd1maxi=0:dd1mini=330:dd2maxi=0:dd2mini=330:dd3maxi=0:dd3mini=0:dd4maxi=0:dd4mini=0
         adjsei=0
 
  residualz=0:varz=0:j=0:mean=0:sq=0
       redim infitz2(itemno)
      redim outfitz2(itemno)
 redim searritem(itemno)

  for j=1 to itemno
                residualz=0:varz=0:outfitran=0
               for jk=1 to personno
                  if isnumeric(test(jk,j))=true then
                     varz=varz+var(jk,j)
                     residualz=residualz+residual(jk,j)^2
                  end if
               next

                outfitran=outfitran+residualz/personno  
            if varz=0 then varz=0.001 
               infitz2(j)= round(residualz/varz  ,2) 
                 mean=mean+infitz2(j)
                 sq=sq+infitz2(j)^2
                 'sd=sd^.05
             if varz>0 then
                searritem(j)=(1/varz)^0.5
             else
                  searritem(j)=1
            end if

   next
 

 
          outfit_mean=round(outfitran/itemno,2)
            mean=mean/itemno
            sd=sq/itemno-mean^2
 
   mmaxi=-110:mmini=100:immaxinfit=0:immininfit=100:immaxoutfit=0:immunoutfit=100
  for j=1 to itemno 

            if item(j)>10 then item(j)=10 
            if item(j)<-10 then item(j)=-10
                limit=item(j)
                outfit=0:residualz=0:varz=0:rawtotal=0
                infitCI=0:infitCI2=0:outfitCI=0 
                 PTMEASUR=0:item_se=0:item_se2=0
                for jk=1 to personno
                   if isnumeric(test(jk,j))=true then
                  outfit=outfit+zscore(jk,j)^2
                  varz=varz+var(jk,j)
                  residualz=residualz+residual(jk,j)^2
                  rawtotal=rawtotal+test(jk,j)

                    infitCI=infitCI + kurtosis(jk, j)
                    infitCI2=infitCI2 + var(jk, j)  
                    outfitCI=outfitCI + kurtosis2(jk, j)
                   PTMEASUR=PTMEASUR+(test(jk,j)-raw_i_avg(j))*(raw_pz(jk)-allscore9_i)
                    item_se=item_se+(test(jk,j)-raw_i_avg(j))^2
                    item_se2=item_se2+(raw_pz(jk)-allscore9_i)^2
            
                  end if
                next


                if item_se=0 or item_se2=0 then
                  PTMEASUR=0
                else
                 PTMEASUR= PTMEASUR/(item_se*item_se2)^0.5 
                end if
                ' objrs3("weight5")=round(PTMEASUR,2)  'weight4= `  
                weight5 =round(PTMEASUR,2)  'weight4= `  
              '  objrs3("threshold")=category_number - 1 
                  'objrs3("a")=  1 
                ' objrs3("c")=  0 
                  weight1 =round(outfit/personno,2) 'outfit
              if verz>0 then
                  weight2 =round(residualz/varz,2) 'infit
                        weight3 =round(1/varz^0.5,2)  
              else
                  weight2=1
                   weight3 =1
              end if

                 ' weight2 =round((residualz/varz+(1-mean)) ,2)
                ' objrs3("weight3")=round(1/varz^0.5,2) 
               ' objrs3("weight1")=round(outfit/personno,2) 'outfit
               ' objrs3("weight2")=round(residualz/varz,2) 'infit
                'objrs3("weight2")=round((residualz/varz+(1-mean)) ,2)
                ' objrs3("weight3")=round(1/varz^0.5,2)  
               
              if rawtotal=0 and zero_item_se>0 then
                ' weight3 =round(1/zero_item_se^0.5,2)
               ' limit =round(zero_item,2)
              elseif rawtotal= personno*1 and large_item_se>0  then
               '  weight3 =round(1/large_item_se^0.5,2)
                ' limit =round(large_item,2)
               end if
         
                For jkm = 0 To category_number - 1 ' mincat 
                    if jkm>0 then
                        mlimit="limit" & jkm
                     
                    end if                              
                          mlimit="answer" & jkm+1
                                     
                next

 

   'response.write   objrs3("limit") & "aa" & objrs3("weight1") & "a " & objrs3("weight2")& "b "&  objrs3("weight3") & "c" & objrs3("weight4") & "d "& objrs3("weight5")
  ' response.end
                ' Sleep(1)
     if varz=0 then varz=.001     
         weight1 =round(outfit/personno,2) 'outfit
               ' objrs3("weight2")=round(residualz/varz,2) 'infit
             if sd>0 then
               weight2 =round((residualz/varz+(1-mean)),2)
              else
                weight2 =round((residualz/varz+(1-mean)),2)
              end if
 

 
   outfitz2(j)= weight1

   if logtrans<>3 and  logtrans<>4 then
     if varz>0 then
     weight2=round(residualz/varz,2)
     else
      weight2=round(residualz/0.001,2)
     end if
   end if
 
   infitz2(j)=weight2
       if varz<=0 then varz=.001
       if maa<=0 then maa=.001
       
             if extremeshifti=1 and vextremeitem>0 and vextremeitem0>0 then  
                    if raw_i(j)=maxcat*personno then 
                       varz= vextremeitem 
                      item_var(j)=varz 
                    end if
                    if raw_i(j)=0 then
                         varz=  vextremeitem0 
                            item_var(j)=varz                       
                      end if
                end if

            
                if varz<=0 then varz=0.01
                weight3 =round(1/varz^0.5,2)
               weight2 =round(residualz/varz,2) 'infit
                if weight3>1.84 then weight3=1.84


        if infitCI>0 and outfitCI>0 and personno-perfect_p>0 then
              InfitCIse=round((InfitCI/InfitCI2^2/(personno))^0.5,2) 
              maa= ((outfitCI/(personno-perfect_p)^2-1/(personno-perfect_p))/(personno-perfect_p)) 
                 if maa>0 then 
                    outfitCIse=round(maa^0.5,2)
                  else
                     outfitCIse=0.001
                   end if
  
                mmark=""

 
            if weight2-2*infitCIse/(personno-perfect_p)^.5 > 1.5 or  weight2+ 2*infitCIse/(personno-perfect_p)^.5 < 0.5 then
                       mmark="<font color=red>*</font>"
               end if
               mmark=""

        if  weight2=0 or  weight1=0 or InfitCIse=0 or OutfitCIse=0 then
           infitzstd=0:outfitzstd=0
         else
           infitzstd=( weight2 ^0.33-1)*(3/(InfitCIse^2*(personno))^0.5)+((InfitCIse^2*(personno))^0.5/3)
           outfitzstd=( weight1^0.33-1)*(3/(OutfitCIse^2*(personno-perfect_p))^0.5)+((OutfitCIse^2*(personno-perfect_p))^0.5/3)
         end if
       else
           infitzstd=0:outfitzstd=0
        end if


  if weight1<0 then weight1=0.01
  if weight2<0 then weight2=0.01
   if weight1>10 then weight1=9.9
  if weight2>10 then weight2=9.9
 'if j=14 then
  'response.write  j &" " & weight1 &" " & weight2 &"<br>"
  'response.end
 'end if
'if  logtrans=4 then 
relia_i=  ((1-exp(weight1)/(1+exp(weight1)))*2)^0.48 
 relia_i2=  ((1-exp(weight2)/(1+exp(weight2)))*2)^0.48  
 'end if
  if request("covid")=""  then 
  %>
<tr><td><%=itemname(j) %>_<%=j %></td><td></td><td><%=round(item(j),2)%></td><td><%=weight3%></td><td><%=round(Aparameter(j),2)%></td><td><%=infitz2(j)%><%=mmark%></td><td><%=weight1%></td><td><%=round(raw_iz(j),2)%></td><td><%=2*infitcise%></td><td><%=2*outfitcise%></td><td><%=round(infitzstd,2)%></td><td><%=round(outfitzstd,2)%></td><td><%=round(weight5,2)%></td><td><%=round(relia_i2,2)%></td><td><%=round(relia_i,2)%>:M=<%=round(mean,2)%></td></tr>
<%  end if



if  request("covid")="41" then 'Slope graphs for person plus outfit
      
   %><h2 id="Copy R code"><H2> Copy R code and pasted to R environment for the plot</h2>
<form action="raschrsm.asp?covid=35" method="post">
<input type="button" value=" Select text" onclick="Geeks()" style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px"><br>
   <p><textarea id="text2" name="remark" rows="25" cols="90" style="background-color:transparent;">
<% response.write contentaa(0) &  chr(13)
For jk = 1 To personno
    arr=split(contentaa(jk),",") 
    mtext = arr(0) 'test(jk, 1)
    For ja = 2 To itemno
        mtext = mtext & "," & arr(ja-1)  'test(jk, ja)
    Next
   arr=split(contentaa(jk),",")
   mname=replace(arr(ubound(arr)-1),",","")
    mtext = mtext & "," & mname & "," & outfitz(jk) & chr(13) 'vbCrLf
    Response.Write mtext 
Next
 
   %> 
  </textarea></p>
</form><div id="bottom"></div>
<%  
 response.end
 end if '41

if request("covid")="36" or request("covid")="37" or request("covid")="39"  or request("covid")="38" or request("covid")="40"  then 'Item Cluster
      
   %><h2 id="Copy R code"><H2> Copy R code and pasted to R environment for the plot</h2>
<form action="raschrsm.asp?covid=35" method="post">
<input type="button" value=" Select text" onclick="Geeks()" style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px"><br>
   <p><textarea id="text2" name="remark" rows="25" cols="90" style="background-color:transparent;">


<%response.write "A1,A2,A3,A4,A5,A6" &chr(13)
 itemmin=100:itemmax=0
for j2=1 to itemno 
    if item(j2)<itemmin then itemmin=item(j2)
     if item(j2)>itemmax then itemmax=item(j2)
      'response.write j2 & itemname2(j2) & chr(13)
next
 'for j2=1 to itemno 
          '  itemno1=itemserial(j2) 
        ' response.write itemno1 & "AA" & round(item(itemno1),2) & "BB" & itemserial(j2) & "CC" & itemname2(itemno1) & chr(13)
  ' response.write itemserial(j2)  & chr(13)
 
 'next
 ' response.end
 redim itemmatrex(itemno,itemno+4)
maxdistance=0:mincorr=100
for j2=1 to itemno
        itemno1=itemserial(j2)
 
        maxcorr=-1 :mcorrsum=0:mindistance2=3330
 for jm=1 to itemno
         itemno2=itemserial(jm) 
 
      if cdbl(itemno1)<>cdbl(itemno2) then 
            mmean1=0: mmean2=0: mmean3=0 
    for jk=1 to personno
        arr=split(contentaa(jk),",")
         if isnumeric(test(jk, itemno1))=false then
               test(jk, itemno1)=0
         end if  
        if isnumeric(test(jk, itemno2))=false then
               test(jk, itemno2)=0
         end if  
     mmean1=mmean1+test(jk,itemno1)
      mmean2=mmean2+test(jk,itemno2)     
    next
  
 mmean1 = mmean1 / personno
    mmean2 = mmean2 / personno
    distance = 0: alldistance = 0
 corr = 0: var1 = 0: var2 = 0: var3 = 0
     For jk = 1 To personno
         corr = corr + (test(jk, itemno1) - mmean1) * (test(jk, itemno2) - mmean2)
          var1 = var1 + (test(jk, itemno1) - mmean1) ^ 2
          var2 = var2 + (test(jk, itemno2) - mmean2) ^ 2
          distance = distance + Abs(test(jk, itemno1) - test(jk, itemno2))
       Next

    
                 If var1 > 0 And var2 > 0 Then
          corr = Round(corr / ((var1 * var2) ^ 0.5), 4)
                 Else
                   corr = 0
                End If
         mcorrsum = mcorrsum + corr
        '  response.write  itemno1 & " " &itemno2 & " " & corr & chr(13)

         If corr > maxcorr Then
           maxcorr = corr: mitemname = "SSS"
          End If
         If distance < mindistance2 Then
            mindistance2 = distance
         End If
         If corr < mincorr Then
           mincorr = corr
          End If
         If distance > maxdistance Then
            maxdistance = distance
          End If
    
   Else  '<>
       corr = 1: distance = 0

    End If '<>

 
     itemmatrex(j2, jm) = corr
 
   Next
          itemmatrex(j2, itemno + 3) = j2
        itemmatrex(j2, itemno + 1) = itemno1  
         itemmatrex(j2, itemno + 2) = mcorrsum
    
    if request("covid")="39" then
        itemmatrex(j2,itemno+2)= round(maxdistance-distance,6)
    elseif request("covid")="38" then
         itemmatrex(j2,itemno+2)=mcorrsum
    elseif request("covid")="40" then
          itemmatrex(j2,itemno+4)=round(maxdistance-distance,6)
    end if
  Next

   

 
 For i = 1 To itemno - 1
  
    For j3 = i + 1 To itemno
             If CDbl(itemmatrex(i, itemno + 2)) > CDbl(itemmatrex(j3, itemno + 2)) Then
               For jm = 1 To itemno + 3
                 SrtTemp = itemmatrex(j3, jm)
                  itemmatrex(j3, jm) = itemmatrex(i, jm)
                 itemmatrex(i, jm) = SrtTemp
               Next
  
             End If
     Next
  Next ' itemno
   
   
     'response.write itemmatrex(1,itemno+3) &  itemmatrex(1,itemno+1) &  itemmatrex(1,itemno+2)
 
    '  response.end
 mtext = ","
For j2 = 1 To itemno
    itemno1 = itemmatrex(j2, itemno + 1)
     wcd = (itemmatrex(j2, itemno + 2))
    
  
            itemnolB = itemno1
    
      no3 = itemmatrex(j2, itemno + 3)
      mitemname1 = "A" & itemnolB
     mvalue = -100
         mitemname = mitemname1
            maxvalue = 0
             kitem = 0
   

  For jm = j2 + 1 To itemno
          
                                                                            
               For mkk = jm To itemno
              
                          
               If CDbl(itemmatrex(j2, mkk)) > maxvalue And itemno1 <> itemserial(mkk) Then
                    
                     If CDbl(itemmatrex(mkk, itemno + 2)) >= CDbl(wcd) Then
                        maxvalue = itemmatrex(j2, mkk)
                       kitem = itemserial(mkk)
             ' response.write kitem & "  :" & mkk & ": " & mtext & wcd & " " & cdbl(itemmatrex(mkk,itemno+2)) & chr(13)
             
                     End If
                    End If
                     
                      If maxvalue <= CDbl(itemmatrex(j2, mkk)) And itemmatrex(j2, mkk) < 1 Then
                         'response.write j2 & " " & itemmatrex(j2, mkk) & "   " & itemno1 & "   " & itemserial(mkk) & "   " & itemmatrex(mkk, itemno + 1) & "   " & wcd & " " & CDbl(itemmatrex(mkk, itemno + 2)) & Chr(13)
                      End If
                     
              Next
         
            If CDbl(kitem) > 0 Then
              mitemname = "A" & kitem
            Else
               mitemname = mitemname1
            End If
 
  Next
         
     mtext = mtext & itemno1 & ","
  If maxvalue < 0 Then maxvalue = 0.01
  If mitemname1 = mitemname Then maxvalue = 1
 
 if request("covid")="36" then
                      response.write replace(itemname2(itemno1)," ","_") & "," & mitemname & "," & round(maxcorr,2) & "," & replace(itemname2(itemno1)," ","_") & "," & round(item(itemno1)-itemmin+.001,2) & "," & infitz2(itemno1) & chr(13)

 elseif request("covid")="37" then
                 response.write replace(itemname2(itemno1)," ","_") & "," & mitemname & "," & round(maxcorr,2) & "," & replace(itemname2(itemno1)," ","_") & "," & round(item(itemno1)-itemmin+.001,2) & "," & round(Zscore(kid,itemno1),2) & chr(13)
 elseif request("covid")="38" and  mitemname1>"" and mitemname>"" then
           response.write mitemname1 & "," & mitemname & "," & Round(maxvalue, 2) & "," & mitemname1 & "," & Round(itemmatrex(j2, itemno + 2) - itemmatrex(1, itemno + 2) + 0.01, 2) & "," & round(item(itemno1)-itemmin+.001,2) & Chr(13)
 
elseif request("covid")="39" then
           response.write mitemname1 & "," & mitemname & "," & Round(maxvalue, 2) & "," & mitemname1 & "," & Round(itemmatrex(j2, itemno + 2) + 0.01, 2) & "," & round(item(itemno1)-itemmin+.001,2) & Chr(13)
elseif request("covid")="40" then
         response.write mitemname1 & "," & mitemname & "," & Round(maxvalue, 2) & "," & mitemname1 & "," & Round(itemmatrex(j2, itemno + 2) - itemmatrex(1, itemno + 2) + 0.01, 2) & "," & Round(itemmatrex(j2, itemno + 4) + 0.01, 2) & Chr(13)
 

 end if
Next
 


 if request("covid")="37" then
          abcd="KID=" & round(person(kid),2) & " Infit=" &round(infitz(kid),2) & " Outfit="& round(outfitz(kid),2) & " SE=" & round(searr(kid),2) 
        ' response.write ",,," & abcd & ",0,0" & chr(13)
          response.write ",,,BB," & round(person(kid)-itemmin+.001,2) & ",0,0" & chr(13)
  end if

  %> 
  </textarea></p>
</form><div id="bottom"></div>
<% 
 
 response.end
 end if '36
 
 
        if weight2>1 then
            adjsei=adjsei+weight2 *weight3*weight3
        else
            adjsei=adjsei+weight3*weight3
        end if 
       

        dd1vari=dd1vari+weight1^2
       dd1meani=dd1meani+weight1
         if  weight1>dd1maxi then dd1maxi=weight1
        if  weight1<dd1mini then dd1mini=weight1
       
        dd2vari=dd2vari+weight2^2
       dd2meani=dd2meani+weight2
         if weight2>dd2maxi then dd2maxi=weight2
        if  weight2<dd2mini then dd2mini=weight2
    
      if  item(j)>mmaxi then  mmaxi=item(j)
      if  item(j)<mmini then  mmini=item(j)
    if infitz2(j)> immaxinfit then immaxinfit=infitz2(j)
     if infitz2(j)< immininfit then immininfit=infitz2(j)
     if outfitz2(j)> immaxoutfit then immaxoutfit=outfitz2(j)
     if outfitz2(j)< imminoutfit then imminoutfit=outfitz2(j) 
 next
  


 if adjsei>0 then
 adjsei=sqr(adjsei/itemno)
 else
 adjsei=1
end if
     ' if adjsei<1 then adjsei=.4
       adjsei=round(adjsei,2)
  dd1meani=round(dd1meani/itemno,2) 
 if (dd1vari/itemno-dd1meani*dd1meani)>0 then
   dd1vari=round(sqr((dd1vari/itemno-dd1meani*dd1meani)),2) 
 else
      dd1vari=1
 end if 
  dd2meani=round(dd2meani/itemno,2) 
 if (dd2vari/itemno-dd2meani*dd2meani)>0 then
   dd2vari=round(sqr((dd2vari/itemno-dd2meani*dd2meani)),2)  
 else
      dd2vari=1
 end if

 if sevar<=0 then sevar=.11
 if sevari<=0 then sevari=.11
 sapi=round(measdi-semeani2,2)
 adjsapi=round(measdi-adjsei*adjsei,2)

 if sapi<=0 then sapi=.001
 if adjsei<=0 then sapi=.001
     
 if adjsapi<=0 then adjsapi=.001
 if semeani2=0 then semeani2=.001
 spmodelGi=round((sapi/semeani2)^.5,2) 
  
 sprealGi=round((adjsapi/(adjsei*adjsei))^.5,2)

 reliamodeli=round(spmodelGi^2/(1+spmodelGi^2),2)
  reliareali=round(sprealGi^2/(1+sprealGi^2),2)
if request("covid")="17" or request("covid")="" then
 
 if score9_var2<=0 then score9_var2=.011
 if measurevar<=0 then measurevar=.011
 if sevar<=0 then sevar=.011
 if sevari<=0 then sevari=.011
 if semeani<=0 then semeani=.011
 

  if maxcat=1 and cdbl(logtrans)=4 then 
     cronbanalpha2=round(mcount2/(mcount2-1)*(1- pqvar/meanrawp2),2)
  elseif  maxcat=1 and cdbl(logtrans)<4 then 
     semean2=modeladjse
      spmodelG=spmodelG2  
      if measurevar-semean2>0 then
       sap=round(measurevar-semean2,2)
      else
       sap=round(measurevar-measurevar/4,2)
      end if
       reliamodel=reliamodel2
 
  end if
 
 
if cronbanalpha2>1 then cronbanalpha2=1  

 

%>
  <table><tr><td>========================================================================</td></tr></table>

<Table>
<tr><td><font color=red>Person(NON-EXTREME)</font> </td><td>RAW_S</td><td>COUNT</td><td>MEAS.</td><td>SE</td><td>Infit</td><td> </td><td>Outfit</td><td> </td></tr>

<tr><td>MEAN </td><td><%=rawmean%></td><td><%=itemno%></td><td><%=measuremean%></td><td><%=round(semean,2)%></td><td><%=dd2mean%></td><td> </td><td><%=dd1mean%></td><td> </td></tr>
<tr><td>S.D. </td><td><%=round(score9_var2^0.5,2)%></td><td>0</td><td><%=round(measurevar^0.5,2)%></td><td><%=round(sevar^0.5,2)%></td><td><%=dd2var%></td><td>  </td><td><%=dd1var%></td><td> </td></tr>
<tr><td>MAX. </td><td><%=rawmax %></td><td><%=itemno%></td><td><%=round(person_max,2)%></td><td><%=round(semax,2)%></td><td><%=dd2max%></td><td> </td><td><%=round(dd1max,2)%></td><td> </td></tr>
<tr><td>MIN.</td><td><%=rawmin%></td><td><%=itemno%></td><td><%=round(person_min,2)%></td><td><%=round(semin,2)%></td><td><%=dd2min%></td><td> </td><td><%=round(dd1min,2)%></td><td> </td></tr>
<tr><td>REAL RMSE(with Infit) </td><td><%=adjse%>    </td><td> ADJ.SD</td><td><%=round(adjsap^0.5,2)%>  </td><td>SEPARATION</td><td><%=sprealG%> </td><td><font color=red>Person</td><td><font color=red>RELIAB.</font></td><td><font color=red><%= reliareal%> </td><td> </td></tr>
<tr><td>MODEL RMSE(mean SE) </td><td><%=round(semean2^0.5,2)%> </td><td> ADJ.SD</td><td><%=round(sap^0.5,2)%> </td><td>SEPERATION</td><td><%=spmodelG%> </td><td>Person</td><td>RELIAB.</td><td><%= reliamodel%> </td><td> </td></tr>
<tr><td>Cronbach's alpha=</td><td><%=cronbanalpha2%></td><td> </td><td>Step delta= </td><td></td><td> </td><td></td><td></td><td> </td><td> </td></tr>
<tr><td>
<%  threshold=category_number-1
     M = threshold + 1
    For Category = 2 To M
      aa= catcalibrate(Category - 1)
      response.write "<td>" & round(aa,2) & "<td>" 
  Next
 

%>
</tr></table>
  <table><tr><td>========================================================================</td></tr></table>
 <Table>
<tr><td><font color=red>Item(NON-EXTREME)</font> </td><td>RAW_S</td><td>COUNT</td><td>MEAS.</td><td>SE</td><td>Infit</td><td> </td><td>Outfit<td><td> </td></tr>

<tr><td>MEAN </td><td><%=rawmeani%></td><td><%=personno%></td><td><%=meameani%></td><td><%=round(semeani,2)%></td><td><%=dd2meani%></td><td> </td><td><%=dd1meani%></td><td> </td></tr>
<tr><td>S.D. </td><td><%=round(rawsdi^0.5,2)%></td><td>0</td><td><%=round(measdi^0.5,2)%></td><td><%=round(sevari^0.5,2)%></td><td><%=dd2vari%></td><td>  </td><td><%=dd1vari%></td><td> </td></tr>
<tr><td>MAX. </td><td><%=rawmaxi %></td><td><%=personno%></td><td><%=round(meamaxi,2)%></td><td><%=round(semaxi,2)%></td><td><%=dd2maxi%></td><td> </td><td><%=round(dd1maxi,1)%></td><td> </td></tr>
<tr><td>MIN.</td><td><%=rawmini%></td><td><%=personno%></td><td><%=round(meamini,2)%></td><td><%=round(semini,2)%></td><td><%=dd2mini%></td><td> </td><td><%=round(dd1mini,2)%></td><td> </td></tr>
<tr><td>REAL RMSE(with Infit) </td><td><%=adjsei%>    </td><td> ADJ.SD</td><td><%=round(adjsapi^0.5,2)%>  </td><td>SEPARATION</td><td><%=sprealGi%> </td><td><font color=red>Item</td><td><font color=red>RELIAB.</font></td><td><font color=red><%= reliareali%> </td><td> </td></tr>
<tr><td>MODEL RMSE(mean SE) </td><td><%=round(semeani2^0.5,2)%> </td><td> ADJ.SD</td><td><%=round(sapi^0.5,2)%> </td><td>SEPERATION</td><td><%=spmodelGi%> </td><td>Item</td><td>RELIAB.</td><td><%= reliamodeli%> </td><td> </td></tr>
 
</table>
    <table><tr><td>========================================================================</td></tr></table>

<% response.end
end if '17

if request("covid")="33" or request("covid")="34" then

     if mmaxp-mminp=0 or mmaxi-mmini=0  then
        response.write "<font color=red>No way due to max=min:" & mmaxp-mminp & "Ai=" & mmaxi-mmini
        response.end
     end if
       if  mmaxinfit-mmininfit=0 or mmaxoutfit-mminoutfit=0 then
        response.write "<font color=red>No way due to max=min: " & mmaxinfit-mmininfit& "Bi=" & mmaxoutfit-mminoutfit
        response.end
     end if
         if immaxinfit-immininfit=0 or immaxoutfit-imminoutfit=0 then
           response.write "<font color=red>No way due to max=min:" & immaxinfit & " " & immininfit & "Ci=" & immaxoutfit-imminoutfit
        response.end
     end if 

  ReDim arr(320)
   if  request("covid")="33" then
      For jk = 1 To personno 
         arr(cdbl(contentaa2(jk))) =cdbl(contentaa2(jk))
      Next 
   else
      arr(0)=0
      personno=itemno
   end if 

  if  request("covid")="33"  then
   redim cbpabc(personno,3)
    for jk=1 to personno 
           cbpabc(jk,2)= contentaa2(jk)
       if request("personm")="1" then     
         cbpabc(jk,1)=personname2(jk) & "_" & round(person(jk),2)        
          cbpabc(jk,3)= round((person(jk)-mminp)/(mmaxp-mminp)*100,2) 
       elseif request("personm")="2" then
                  cbpabc(jk,1)=personname2(jk) & "_" & round(outfitz(jk),2)        
          cbpabc(jk,3)= round((outfitz(jk)-mminoutfit)/(mmaxoutfit-mminoutfit)*100,2) 
       
             else
             cbpabc(jk,1)=personname2(jk) & "_" & round(infitz(jk),2)        
          cbpabc(jk,3)= round((infitz(jk)-mmininfit)/(mmaxinfit-mmininfit)*100,2) 
            end if

    next
  else
 
     redim cbpabc(itemno,3)
    for jk=1 to itemno
         cbpabc(jk,2)= 0
        if request("itemd")="1" then    
           cbpabc(jk,1)=itemname(jk) & "_" & round(item(jk),2) 
          cbpabc(jk,3)= round((item(jk)-mmini)/(mmaxi-mmini)*100,2) 
        elseif request("itemd")="2" then
         cbpabc(jk,1)=itemname(jk) & "_" & round(outfitz2(jk),2) 
          cbpabc(jk,3)= round((outfitz2(jk)-imminoutfit)/(immaxoutfit-imminoutfit)*100,2) 
        else
          cbpabc(jk,1)=itemname(jk) & "_" & round(infitz2(jk),2) 
          cbpabc(jk,3)= round((infitz2(jk)-immininfit)/(immaxinfit-immininfit)*100,2) 
        end if 
    next
  end if
 
  groupnumber = 0
  For j = 0 To UBound(arr)
     If arr(j) > "" Then
       groupnumber = groupnumber + 1
     End If
  Next 
    redim arrname(3)
    dataframe = "": arrname(1)="individual":arrname(2)="group":arrname(3)="value"
   For j = 1 To 3
       colname = arrname(j)    
      mtext = ""
     For jk = 1 To personno
       If jk = 1 Then        
         If j <= 2 Then
           mtext = Chr(34) &  cbpabc(jk,j) & Chr(34)
         Else
          mtext = cbpabc(jk,j)
         End If
       Else
        If j <= 2 Then
           mtext = mtext & "," & Chr(34) &  cbpabc(jk,j) & Chr(34)
        Else
         mtext = mtext & "," & cbpabc(jk,j)
        End If
       End If
     Next  
      If j <= 1 Then
      dataframe = arrname(j)  & "=c(" & mtext & ")"
      Else
         dataframe = dataframe & ", " & arrname(j)  & "=c(" & mtext & ")"
        End If
    Next  
    dataaaa1 = "data <- data.frame(" & dataframe & ")" & chr(13)
     
   hjust = "1,1,0,0"
    hjust = "1"
     For j = 2 To groupnumber
      hjust = hjust & ",0"
    Next  
     groupnumber = 4
     dataaaa2 = "annotate(" & Chr(34) & "text" & Chr(34) & ", x = rep(max(data$id)," & groupnumber & "), y = c(20, 40, 60, 80), label = c(" & Chr(34) & "20" & Chr(34) & "," & Chr(34) & "40" & Chr(34) & "," & Chr(34) & "60" & Chr(34) & "," & Chr(34) & "80" & Chr(34) & ") , color=" & Chr(34) & "red" & Chr(34) & ", size=3 , angle=0, fontface=" & Chr(34) & "bold" & Chr(34) & " , hjust=1) +"  & chr(13)
     dataaaa3  = "geom_text(data=base_data, aes(x = title, y = -18, label=group), hjust=c(" & hjust & "), colour = " & Chr(34) & "red" & Chr(34) & ", alpha=0.98, size=4, fontface=" & Chr(34) & "bold" & Chr(34) & ", inherit.aes = FALSE)"  & chr(13)
   
   %><h2 id="Copy R code"><H2> Copy R code and pasted to R environment for the plot</h2>
<form action="raschrsm.asp?covid=33" method="post">
<input type="button" value=" Select text" onclick="Geeks()" style="font-size:12pt;color:white;background-color:green;border:2px solid #336600;padding:3px"><br>
   <p><textarea  id="text2" name="remark" rows="25" cols="90" style="background-color:transparent;">       
library(tidyverse) 
# Create dataset
<%=dataaaa1%>
# Set a number of 'empty bar' to add at the end of each group
empty_bar <- 3
to_add <- data.frame( matrix(NA, empty_bar*nlevels(data$group), ncol(data)) )
colnames(to_add) <- colnames(data)
to_add$group <- rep(levels(data$group), each=empty_bar)
data <- rbind(data, to_add)
data <- data %>% arrange(group)
data$id <- seq(1, nrow(data))

# Get the name and the y position of each label
label_data <- data
number_of_bar <- nrow(label_data)
angle <- 90 - 360 * (label_data$id-0.5) /number_of_bar     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
label_data$hjust <- ifelse( angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle+180, angle)
 
# prepare a data frame for base lines
base_data <- data %>% 
  group_by(group) %>% 
  summarize(start=min(id), end=max(id) - empty_bar) %>% 
  rowwise() %>% 
  mutate(title=mean(c(start, end)))
 
# prepare a data frame for grid (scales)
grid_data <- base_data
grid_data$end <- grid_data$end[ c( nrow(grid_data), 1:nrow(grid_data)-1)] + 1
grid_data$start <- grid_data$start - 1
grid_data <- grid_data[-1,]
 
# Make the plot
p <- ggplot(data, aes(x=as.factor(id), y=value, fill=group)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
  
  geom_bar(aes(x=as.factor(id), y=value, fill=group), stat="identity", alpha=0.5) +
 
  # Add a val=100/75/50/25 lines. I do it at the beginning to make sur barplots are OVER it.
  geom_segment(data=grid_data, aes(x = end, y = 80, xend = start, yend = 80), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 60, xend = start, yend = 60), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 40, xend = start, yend = 40), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 20, xend = start, yend = 20), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 10, xend = start, yend = 10), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  # Add text showing the value of each 100/75/50/25 lines
 <%=dataaaa2%>
 
  geom_bar(aes(x=as.factor(id), y=value, fill=group), stat="identity", alpha=0.5) +
  ylim(-100,120) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-1,4), "cm") 
  ) +
   coord_polar() + 
    geom_text(data=label_data, aes(x=id, y=value+10, label=individual, hjust=hjust), color="black", fontface="bold",alpha=0.98, size=3.5, angle= label_data$angle, inherit.aes = FALSE ) +
    
    # Add base line information
    geom_segment(data=base_data, aes(x = start, y = -5, xend = end, yend = -5), colour = "black", alpha=0.98 , size=1.6 , inherit.aes = FALSE )  +
 <%=dataaaa3%>
p
     </textarea></p>

    <p><input type="submit" value="Submit"></p>
  </select>
</form><div id="bottom"></div>
<%    
 response.end
 end if 
 

 %>
 <table>
 

 <tr><div><h4>  STRUCTURE-THRESHOLD MEASURE ANCHOR FILE FOR LIKING FOR SCIENCE (Wright & Masters p.18)<br>
            CATEGORY  Rasch-Andrich threshold</h4></div>
<%  
     
          For jkm = 1 To category_number - 1 ' mincat
              response.write  jkm  & ".=" & round(catcalibrate(jkm),2)  &"<BR>" 'step(jkm)     ' using original step difficulties
         Next 
 
  if request("covid")="14" then
     response.end
  end if
 '=====================================================================
  if  request("covid")="15" or kano="YES"  or request("covid")="16" or request("covid")="18" then  'Kanoplot
 

%>
<form id="form1" method="post" action="../kpiall/kanoplot.asp"  name="post"  >
  
:<textarea rows="15" name="content" cols="60">
<%   
      
     for jk=1 to mrow
               rkk2=instr("ABCDEFG",personrk(jk))
         if  CDbl(request("categoryabc"))=1 then
           response.write "Obs." & jk & "	" & round(test2(jk,1),2)  & "	" & round(test2(jk,2),2)  & "	" & round(person(jk)*10+100,2) & "	" & round(outfitabc(jk),2)  & chr(13)  
              xname="Variable2":yname="Variable1"   
          else 
          if request("covid")="15" then 
               xname="Outfit" :yname="Measure"     
           response.write "Obs." & jk & "	" & round(person(jk),2)  & "	" & round(outfitabc(jk),2)   & "	" & round(person(jk)*10+100,2) & "	" & round(rkk2,2)  & chr(13)  
          elseif request("covid")="16" then
                xname="Infit":yname="Measure"  
      response.write "Obs." & jk & "	" & round(person(jk),2)  & "	" & round(infitz(jk),2)   & "	" & round(person(jk)*10+100,2) & "	" & round(rkk2,2)  & chr(13)  
          elseif request("covid")="18" then
          xname="Measure":yname="Rawscore"  
            response.write "Obs." & jk & "	" &  raw_p2(jk)   & "	" & round(person(jk),2)   & "	" & round(person(jk)*10+100,2) & "	" & round(rkk2,2)  & chr(13)  
       
         end if
          end if
  next
 


%>
   </textarea><br>
           Name, citation, publication, and x-index, for example,  with blanks from MS Excel using copy and pasted methods and bubble size is the person measures
<br>
        <input type="hidden" name="xname" value=<%=xname%>> 
 <input type="hidden" name="covid" value=<%=covid%>>
         <input type="hidden" name="yname" value=<%=yname%>>  
       X-axis: <input type="text" name="xaxis" value=35 style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
       Y-axis: <input type="text" name="yaxis" value=16 style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
       Move forward on X: <input type="text" name="addx" value=0 style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
       Move forward on Y: <input type="text" name="addy" value=0 style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
      Bubble: <input type="text" name="bubblesize" value=1 style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>   
Wider on X: <input type="text" name="addx2" value=1 style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
   Wider on Y: <input type="text" name="addy2" value=1 style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>  
           <input type="submit" value="Submit" style="width:500px;font-size:13pt;padding:2px; border:3px solid green"><BR>
      <input type="button" name="button" value="Read me" onClick="location.href='../article/article16/DownloadPubmeddataandcitations.pdf'" style="width:500px;font-size:13pt;padding:2px; border:3px solid green">
    <input type="button" name="button" value="Forest plot" onClick="location.href='forestplot.asp'" style="width:500px;font-size:13pt;padding:2px; border:3px solid green"><br> 

    <textarea rows="15" name="content2" cols="60">
27.1694527	-200	-68.02897986	-200
27.67208952	-190	-67.5292349	-190
28.17472635	-180	-67.02948993	-180
28.67736317	-170	-66.52974497	-170
29.18	-160	-66.03	-160
29.68263683	-150	-65.53025503	-150
30.37828245	-140	-62.5118935	-140
31.0739925	-130	-57.49346235	-130
31.7697854	-120	-54.47494169	-120
32.46568876	-110	-47.45630169	-110
33.16172099	-100	-41.43752246	-100
33.85793733	-90	-35.41854432	-90
34.5544022	-80	-29.39929766	-80
35.25120765	-70	-23.37968303	-70
35.948547	-60	-17.35949157	-60
36.64672401	-50	-11.3383951	-50
37.346337	-40	-5.315747176	-40
38.04879436	-30	0.709973821	-30
41.63834157	-20	8.758146296	-20
45.7908829	-10	9.493207933	-10
50.8059296	0	11.02524267	0
53.80846377	10	10.69767351	10
57.9761373	20	11.44671761	20
62.1036343	30	12.15863776	30
66.22185672	40	12.861988	40
69.33648289	50	13.56201523	50
72.44934578	60	14.26041316	60
74.56121248	70	14.95789058	70
76.67245158	80	15.65478809	80
78.9	90	16.35130819	90
80.52754842	100	17.04757054	100
82.35509683	110	17.7436396	110
83.18264525	120	18.43956137	120
84.41019367	130	19.13537268	130
85.43774208	140	19.83110114	140
85.9652905	150	20.52675597	150
86.49283892	160	21	160
86.72038733	170	21.47324403	170
86.94793575	180	21.94648807	180
87.17548416	190	22.4197321	190
87.40303258	200	22.89297613	200
</textarea><br>
           This is used for plot the Kano diagram
<br>  
  </form>   
   </center>   

<%  response.end
   end if  '=15  16
    
'===================================

if request("groupabc")="1" or  request("covid") ="06"  or request("covid")="07" then
 
      ' contentaa2=split(content2,chr(13))



        redim persondif(1)
            mgender=""
            redim persondif(1)
 
           for jk=1 to personno
             contentaa2(jk)=trim(contentaa2(jk))
            contentaa2(jk)=replace(contentaa2(jk), chr(10),"")
           contentaa2(jk)=replace(contentaa2(jk), chr(13),"")
          contentaa2(jk)=replace(contentaa2(jk), chr(12),"")
         contentaa2(jk)=replace(contentaa2(jk), chr(11),"")
         if mgender="" and contentaa2(jk)>"" then 
               mgender=contentaa2(jk)
         elseif contentaa2(jk)>"" then
               mgender=mgender & "," & contentaa2(jk)
         end if
' response.write "bbb" & mid(contentaa2(jk),1,1) &" " & asc(mid(contentaa2(jk),1,1))
 
          if   mid(contentaa2(jk),1,1) =0 then
            persondif(0)=persondif(0)+1
         else
                persondif(1)=persondif(1)+1
         end if
  

       next
   

end if


'============================
 if (request("covid"))="06"  or (request("covid"))="07" then


 
   'ReDim test(personno, itemno)
  
   ' ReDim Item(itemno)
  ' ReDim catcalibrate(categoryabc + 1)
    ReDim  catstep(1, categoryabc+1) 
    maxcat = categoryabc + 1
    mincat = 0
    itemno = itemno
    '   catcalibrate(0) = 0
           
        mgender= "0,1"           
  kgender=split( mgender,",")
   kgender(0)=0
  kgender(1)=1
maxcat =categoryabc  
mincat=0 
      ReDim personmea(1, personno)
      ReDim itemdif(1, itemno), itemdifse(1, itemno)
     ReDim difraw(UBound(kgender), itemno), difexp(UBound(kgender), itemno)
      
 For iteration2 = 0 To UBound(kgender)
     personnodif = persondif(iteration2)
 ' redim item(itemno),person(personnodif),catcalibrate(maxcat-mincat+1)
  ReDim catresi(maxcat - mincat + 1), catthresh(maxcat - mincat + 1), catadj(maxcat - mincat + 1)
  ReDim expect(personnodif, itemno), Var(personnodif, itemno), residual(personnodif, itemno)
  category_number =categoryabc+1
     ReDim catobs(categoryabc + 1), catexp(categoryabc + 1)
      personmin2=999: personmax2=0       
     allscore9 = 0:knum=1
       For jk =1 To personno 
  
        If  CDbl(contentaa2(jk)) =  CDbl(iteration2) Then
             
            if maxcat*itemno<>raw_p(jk) and raw_p(jk)> personmax2 then
               personmax2=raw_p(jk)
             end if
             if 0<>raw_p(jk) and raw_p(jk)<personmin2 then
                  personmin2=raw_p(jk)
            end if

          For J =1 To itemno  
              if isnumeric(test(jk, j))=true then
             ma = test(jk , J )
            catobs(ma) = catobs(ma) + 1
              end if
         Next  
        End If
      Next 

       redim raw_i2(itemno) 
           for j=1 to itemno
             for jk=1 to personno
              If  CDbl(contentaa2(jk)) =  CDbl(iteration2) Then
                    If IsNumeric(test(jk, J)) = True Then
                        raw_i2(j)=raw_i2(j)+ test(jk, j)
                    end if
              end if
             next
           next
    itemmin2=999: itemmax2=0 
 for j=1 to itemno
            
        if maxcat*persondif(iteration2)<>raw_i2(j) and raw_i2(j)> itemmax2 then
              itemmax2=raw_i2(j)
        end if
        if 0<>raw_i2(j) and raw_i2(j)< itemmin2 then
                  itemmin2=raw_i2(j)
        end if
 next
     category_number=categoryabc+1
ReDim raw_pz(personnodif), raw_iz(itemno)
        mnum = 1
  
 ReDim expect(personnodif, itemno), Var(personnodif, itemno), Zscore(personnodif, itemno), residual(personnodif, itemno)
   
     perfect_i = 0: perfect_p = 0
 iterat = 40
 If iterat = "" Then iterat = 30


 redim  catcalibrate(maxcat-mincat+1)
 For Iteration = 1 To iterat
         zscore_mean = 0: zscore_sd = 0
   
   ReDim item_error(itemno), item_var(itemno)
        resi_a = 0: resi_b = 0
        sumsqerror = 0
   ReDim Var_P(personnodif), person_exp(personnodif)

             For J = 1 To itemno
                      difraw(iteration2, J) = 0
                      difexp(iteration2, J) = 0
             Next


   mnumk = 1
   For jk = 1 To personno
            person_error = 0: person_max = -100: person_min = 100
            sqaure_resi = 0
            
     If  CDbl(contentaa2(jk)) =  CDbl(iteration2) Then

       For J = 1 To itemno
     If IsNumeric(test(jk, J)) = True Then
      logit = personmea(iteration2, mnumk) - itemdif(iteration2, J)
       normalizer = 0 ' this will force the sum of the probabilities = 1.0
      sumsqu = 0
      currentlogit = 0: all_asum = 0
     ReDim expaaa(category_number - 1)
     catcalibrate(0) = 0 
    
   For cata = 0 To categoryabc 
            msum_tau = 0
          If category_number > 2 Then
              For jk2 = 0 To cata
                  msum_tau = msum_tau + catcalibrate(jk2)
              Next
           End If
           expaaa(cata) = Exp(cata * logit - msum_tau)
        all_asum = all_asum + expaaa(cata)     
  Next
             
    exp_a = 0: kurtosisZ = 0
     For cata = 0 To categoryabc 
       exp_a = exp_a + cata * expaaa(cata) / all_asum              
     Next
    
       var_a = 0
       For cata = 0 To  categoryabc
        kurtosisZ = kurtosisZ + (exp_a - cata) ^ 4 * expaaa(cata) / all_asum
        var_a = var_a + (exp_a - cata) ^ 2 * expaaa(cata) / all_asum
        catexp(cata) = catexp(cata) + expaaa(cata) / all_asum
      Next
  
             expect(mnumk, J) = exp_a
             Var(mnumk, J) = var_a
           
             residual(mnumk, J) = test(jk, J) - exp_a
              person_error = person_error + residual(mnumk, J)
              Var_P(mnumk) = Var_P(mnumk) + var_a
   if  jk=4 then
' response.write  jk & ".  " & test(jk, J) & " " & exp_a & " " & round(residual(mnumk, J)  ,4) & " " & round(var_a,3)  & "<br>" 
    end if
              item_var(J) = item_var(J) + var_a
                difraw(iteration2, J) = difraw(iteration2, J) + test(jk, J)
                 difexp(iteration2, J) = difexp(iteration2, J) + expect(mnumk, J)
              
              item_error(J) = item_error(J) + residual(mnumk, J)
             ' Zscore(jk, j)  =  residual(jk,j)/var(mnumk,j)^0.5
              sqaure_resi = sqaure_resi + residual(mnumk, J) ^ 2
              zscore_mean = zscore_mean + Zscore(mnumk, J)
                zscore_sd = zscore_sd + Zscore(mnumk, J) ^ 2
     End If
 
      Next 'item


           resi_a = resi_a + person_error
        If Iteration = 1 And jk >= 1 And iteration2 = 1 Then
            ddfd = 1
        End If
      If Abs(person_error) > sumsqerror Then
            sumsqerror = Abs(person_error)
      End If
  
         If Var_P(mnumk) < 0.0001 Then Var_P(mnumk) = 0.0001
   
          If person_error/ Var_P(mnumk) > 10 Then
            personmea(iteration2, mnumk) = 10
          ElseIf person_error/ Var_P(mnumk) < -10 Then
                 personmea(iteration2, mnumk) = -10
          Else
               personmea(iteration2, mnumk) = personmea(iteration2, mnumk) + person_error/ Var_P(mnumk)
          End If
                 
        
        
              if raw_p(jk)=personmax2 then personmax2m=personmea(iteration2, mnumk)
               if raw_p(jk)=personmin2 then personmin2m=personmea(iteration2, mnumk) 
              if personmea(iteration2, mnumk)=10 or personmea(iteration2, mnumk)=-10 then extremeshiftp=1
       
               mnumk = mnumk + 1
    
     End If ' if contentaa2(iteration2)=iteration2 then
   Next 'person
      ' response.write "AAAAA" & mnumk
'response.end
               item_avg = 0: extremescore = 0
          For J = 1 To itemno
           If Abs(item_error(J)) > sumsqerror Then
              '  sumsqerror = Abs(item_error(J))
                        
            End If
                resi_b = resi_b + item_error(J)
 
            If item_var(J) < 0.0001 Then item_var(J) = 0.0001
            If item_error(J) / item_var(J) > 10 Then
              itemdif(iteration2, J) = 10
            ElseIf item_error(J) / item_var(J) < -10 Then
              itemdif(iteration2, J) = -10
            Else
                itemdif(iteration2, J) = itemdif(iteration2, J) - item_error(J) / item_var(J)
            End If
            If item_var(J) = 0.0001 Then
                itemdif(iteration2, J) = 0 'Item(j)
                ' extremescore=extremescore+1
            End If
            itemdifse(iteration2, J) = Round(Sqr(1 / item_var(J)), 2)
                item_avg = item_avg + itemdif(iteration2, J)
             if raw_i2(j)=itemmax2 then itemmax2m= itemdif(iteration2, j)
               if raw_i2(j)=itemmin2 then itemmin2m= itemdif(iteration2, j) 
              if  itemdif(iteration2, j)=10 or  itemdif(iteration2, j)=-10 then extremeshiftp=1
          Next
 
           item_avg = item_avg / itemno
            For J = 1 To itemno
              itemdif(iteration2, J) = itemdif(iteration2, J) - item_avg
                If item_var(J) = 0.0001 Then
                   Itemdif(iteration2, J) = 0
                End If
              '   Sheets("dd").Cells(1, J + 1) = itemdif(iteration2, J)
            '    response.write  "<br>" & j &  " " & round(item(j),2) & ":" & round(item_avg,2)
             Next
          
           cat_avg = 0
                mmin = 1000: mmax = -3000
          For jkm = 0 To category_number - 1 ' mincat
               'catexp
                catresi(jkm) = catobs(jkm) - catexp(jkm)
                  If catresi(jkm) < mmin Then
                     mmin = catresi(jkm)
                   ElseIf catresi(jkm) > mmax Then
                       mmax = catresi(jkm)
                   End If
                If Abs(catresi(jkm)) > sumsqerror Then
                  ' sumsqerror = Abs(catresi(jkm))
               End If
              If jkm > 0 Then
                If catobs(jkm) > 0 And catexp(jkm) > 0 Then
                 catthresh(jkm) = catcalibrate(jkm) + Log(catobs(jkm - 1) / catobs(jkm)) - Log(catexp(jkm - 1) / catexp(jkm))
                Else
                  catthresh(jkm) = catcalibrate(jkm)
          
                 End If
                cat_avg = cat_avg + catthresh(jkm)
              Else
                  catthresh(jkm) = 0
              End If
           '  Sheets("dd").Cells(10, jkm + 4) = catthresh(jkm)
          Next
                    If Abs(mmax) > Abs(mmin) Then
                          catzz = Abs(mmax)
                    Else
                         catzz = Abs(mmin)
                    End If
                 cat_avg = cat_avg / (category_number - 1)
             catadj(0) = 0
          For jkm = 1 To category_number - 1 ' mincat
                catadj(jkm) = catthresh(jkm) - cat_avg
              catcalibrate(jkm) = catadj(jkm)
              catstep(iteration2, jkm) = catadj(jkm)
            '   response.write jkm & ".  " & round(catadj(jkm) ,4)  & "  " & catobs(jkm) & "  " & catexp(jkm)  &"<br>"       
          Next
       If sumsqerror < 0.05 Or Abs(sumsqerror2 - sumsqerror) < 0.01 Then
           Exit For
       End If
         sumsqerror2 = sumsqerror
      
  Next 'iteration


          ' for extreme scores
      if extremeshiftp=1 then
       extremeperson=(personmax2m-personmin2m)/(personmax2-personmin2)*(maxcat*itemno-personmax2)+personmax2m
      extremeperson0=(personmax2m-personmin2m)/(personmax2-personmin2)*(0-personmin2)+personmin2m
 
     
    for jk2=1 to persondif(iteration2)
        if personmea(iteration2, jk2)=10  then personmea(iteration2, jk2)=extremeperson 
        if personmea(iteration2, jk2)=-10 then personmea(iteration2, jk2)=extremeperson0 
      
     next
      end if
      if extremeshifti=1 then      
        extremeitem=(itemmax2m-itemmin2m)/(itemmax2-itemmin2)*(maxcat*personno-itemmax2)+itemmax2m
        extremeitem0=(itemmax2m-itemmin2m)/(itemmax2-itemmin2)*(0-itemmin2)+itemmin2m
      
        for j=1 to itemno 
        if itemdif(iteration2, j)=10 then itemdif(iteration2, j)=extremeitem 
        if itemdif(iteration2, j)=-10 then itemdif(iteration2, j)=extremeitem0 
      next
      end if 



Next 'iteration2


 %>
<Table><table><tr><td>Thresholds</td><td>Group 1</td><td>Group 2</td></tr><tr>
<% 
           For jkm = 1 To categoryabc ' mincat             persondif(0)
                
        response.write "<td>Step" & jkm & "</td><td>" & round(catstep(0, jkm),2) &"</td><td>" & round(catstep(1, jkm),2) &"</td><tr>" 
           next %>
           </Table>          
     <%      
 

   if request("covid")="07" then

%>
<form id="form1" method="post" action="../kpiall/QSubgrouptest.asp"  name="post"  >
          
            responses(rows for each entity)<br>
:<textarea rows="15" name="content" cols="60">
<%
     
               n1 = persondif(0)
               n2 = persondif(1)
             if n1<n2 then
                df=n1-1
             else
                 df=n2-1
              end if
       redim Weightk(itemno), vark(itemno)
               totalvark=0
          for j=1 to itemno
              redim arr(4)
              arr(0)=itemdif(0,j)
               arr(2)=itemdif(1,j)
              arr(1)=itemdifse(0,j)*sqr(n1)
               arr(3)=itemdifse(1,j)*sqr(n2)
  
             sd = (n1 - 1) * arr(1) ^ 2 + (n2 - 1) * arr(3) ^ 2
             sd = sd / (n1 + n2 - 2)
             sd = Sqr(sd)
               if sd=0 then sd=1
             Cohend = (arr(0) - arr(2)) / sd
              Hedgesg = Cohend
             vard = (n1 + n2) / (n1 * n2) + Cohend * Cohend / (2 * (n1 + n2))
          
             Jcorrect = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
             Hedgesg = Cohend * Jcorrect
             Varg = vard * Jcorrect * Jcorrect + Vartou
              sd = Sqr(Varg)
                variance = sd * sd
                totalvark= totalvark+ 1 / sd * 1 / sd
           next 
    mvar = 0: meany = 0: Qsqure = 0: wsqure = 0:: wsqure3 = 0
 for j=1 to itemno

             redim arr(4)
              arr(0)=itemdif(0,j)
               arr(2)=itemdif(1,j)
              arr(1)=itemdifse(0,j)*sqr(n1)
               arr(3)=itemdifse(1,j)*sqr(n2)
  
             sd = (n1 - 1) * arr(1) ^ 2 + (n2 - 1) * arr(3) ^ 2
             sd = sd / (n1 + n2 - 2)
             sd = Sqr(sd)
             if sd=0 then sd=1
             Cohend = (arr(0) - arr(2)) / sd
              Hedgesg = Cohend
             vard = (n1 + n2) / (n1 * n2) + Cohend * Cohend / (2 * (n1 + n2))
      
             Jcorrect = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
             Hedgesg = Cohend * Jcorrect
             Varg = vard * Jcorrect * Jcorrect + Vartou
              sd = Sqr(Varg)
              if sd=0 then sd=1
             measure = Hedgesg
             SE = sd
             variance = sd * sd
           if sd>0 and totalvark>0 then
             weight  = round((1 / sd * 1 / sd)/ totalvark*100,2)
           else
              weight  =1
            end if
             Zvalue=round(Hedgesg/sd, 2)
            strSQL =  "Select * From ttest2  where tvalue=" & abs(round(Zvalue,2))
                Set objRS = GetSQLRecordset(strSQL, "../News.mdb", "ttest2")
             if objrs.eof then
                pvalue=0.001
             else
              df2=df
            if df>50 then df2=50
                 dfdd="df" & df2
               tprob=objrs(dfdd)
              pvalue=round(tprob*2,3)  
             end if  
              lowlimit= Round(measure - SE * 1.96, 3)
              uplimit= Round(measure + SE * 1.96, 3)
           
 
          mtext="No." & j & ". Groups(0 vs. 1)," &  measure & ","  & SE & ","  & lowlimit  & ","  & uplimit & ","  & df  & ","  & Zvalue  & ","  & pvalue  & ","  &  weight & chr(13)
           response.write mtext


         mvar = mvar + 1 / variance  'W
        meany = meany + 1 /  variance * measure 'WY
         Qsqure = Qsqure + 1 /  variance * measure*measure 'WYY
         wsqure = wsqure + (1 /  variance) * (1 / variance) 'WW
        wsqure3 = wsqure3 + (1 /  variance) * (1 / variance) * (1 /  variance) 'WW2
next 
     Vartou=0
   
     ' For jk = 1 To ITEM
       
    '  Next  
      If Vartou = 0 Then 'fixed model
        '  Sheets("data").Cells(15, "AK") = mvar
       '  Sheets("data").Cells(15, "AN") = wsqure
       '  Sheets("data").Cells(15, "AO") = wsqure3
        '  Sheets("data").Cells(15, "AL") = meany
        '  Sheets("data").Cells(15, "AM") = Qsqure
     End If
      Qsqure = Qsqure - meany * meany / mvar
      cquare = mvar - wsqure / mvar

      If cquare = 0 Then
          Tauquare = 0
       Else
        Tauquare = Round((Qsqure - (lastrows2 - 3)) / cquare, 14)
       End If
        '=================
       meany = Round(meany / mvar, 3)
     
        mvar3 = mvar
     
        isquare = 0
        For jk = 1 To itemno
             isquare = isquare + 1 / variance * (measure - meany) ^ 2
       Next  
                     

     
           
               pvalue= Round(Sqr(1 / mvar), 3)
               uplimit = Round(meany + 1.96 * pvalue, 3)
               lowlimit = Round(meany - 1.96 *pvalue, 3)
               Zvalue=Round(meany / Sqr(1 / mvar3), 3)
                 measure=Round((meany), 3)
                  SE= Round(Sqr(1 / mvar3), 3)
               variance = Round(1 / mvar, 3)
  
            df=persondif(0)+persondif(1)-2
          strSQL =  "Select * From ttest2  where tvalue=" & abs(round(Zvalue,2))
                Set objRS = GetSQLRecordset(strSQL, "../News.mdb", "ttest2")
             if objrs.eof then
                pvalue=0.001
             else
             df2=df
              if df>50 then df2=50
                 dfdd="df" & df2
               tprob=objrs(dfdd)
              pvalue=round(tprob*2,3)  
             end if  
   mtext="    Overall," &  measure & ","  & SE & ","  & lowlimit  & ","  & uplimit & ","  & df  & ","  & Zvalue  & ","  & pvalue  & ",100" & chr(13)
             mtext=replace(mtext, chr(10),"")
            
             response.write mtext 
 %>
 </textarea><br>
<%
kscale=1
Towardtheright=0
criterio=0
if criterio="" then criterio=1
if kscale="" then kscale=0
if Towardtheright="" then Towardtheright=0
if Twoside="" then Twoside=0
if multiplyratio="" then multiplyratio=1
if TExtreme="" then TExtreme=1
%>
            Group if necessary from 1 to n at least 5 ovserved number for each group)<br>
 
      EffectSize: <input type="text" name="kscale" value=<%=kscale%> style="width:50px;font-size:13pt;padding:2px; border:3px solid green">
      Scale(>0;eg 0.9 or 1) 
<input type="text" name="criterio" value=<%=criterio%> style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
       Toward the right(>0) <input type="text" name="Towardtheright" value=<%=Towardtheright%> style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
       Multiply a ratio on scale(<=1) <input type="text" name="multiplyratio" value=<%=multiplyratio%> style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4><BR>
 Extended to Two sides(<>0) <input type="text" name="Twoside" value=<%=Twoside%> style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4>
Extreme=<input type="text" name="TExtreme" value=<%=TExtreme%> style="width:50px;font-size:13pt;padding:2px; border:3px solid green" size=4><br>
       <input type="submit" value="Submit" style="width:500px;font-size:13pt;padding:2px; border:3px solid green">  
           
  </form>   
   </center>  
<%  response.end
   
    end if '=7
           
 
  if ubound(kgender)=1 then   
%>
 <table><tr>DIF class/group specification is: <font color=red>Pairwise DIF=<%=mgender%>(1)(If an item has a perfect socre, the delta is assigned with an overall delta) </font></tr>
 
<tr><td>  KID  </td><td>Obs-Exp</td><td>DIF </td><td>DIF</td><td>KID</td><td>Obs-Exp </td><td> DIF </td><td>DIF</td><td> DIF</td><td> JOINT</td><td>Rasch-</td><td>Welch</td><td>  Item</td><td> </td></tr>                               
<tr><td>  CLASS</td><td>Average</td><td>MEASURE</td><td> S.E.</td><td> CLASS</td><td> Average</td><td>MEASURE</td><td> S.E. </td><td>  CONTRAST </td><td>S.E.</td><td> t </td><td>sig.</td><td>df</td><td>  Name</td></tr>                         
 <% j=1

   'response.write persondif(jm)
   for j=1 to itemno 
    
               tvalue2=""
                 see2=round(sqr(itemdifse(0,j)^2+itemdifse(1,j)^2),2)
               tvalue=round((itemdif(0,j)-itemdif(1,j))/see2,2)
 
 if persondif(0)<persondif(1) then
     df=persondif(0)-1
 else
     df=persondif(1)-1
 end if
if df="" then df=10
if tvalue="" then tvalue=0
if abs(tvalue)>5 then tvalue=5
if df>100 then df=100
 

   strSQL =  "Select * From ttest2  where tvalue=" &  abs(round(tvalue,2)) 
   Set objRS = GetSQLRecordset(strSQL, "../News.mdb", "ttest2")
 if objrs.eof then
     
  tvalue2= " <font color=red>" & round(0.001,3) &"</font>"
 else
  df2=df
    if df>50 then df2=50
  dfdd="df" & df2
 
     tprob=objrs(dfdd)
    if tprob*2<0.05 then
         tvalue2= " <font color=red>" & round(tprob*2,3) &"</font>"
   else
           tvalue2=round(tprob*2,3) 
    end if
 end if
           '  if abs(tvalue)>=2 then tvalue2="sig"
                      

   jm=0
  %>
<tr align=center><td><%=kgender(0)%></td>
<td><%=round(difraw(0,j)/persondif(0)-difexp(0,j)/persondif(0),2)%></td>
<td><%= round(itemdif(0,j),2) %></td>
<td><%=round(itemdifse(0,j),2)%></td>
<td><%=kgender(1)%></td>
<td><%=round(difraw(1,j)/persondif(1)-difexp(1,j)/persondif(1),2)%></td>
<td><%= round(itemdif(1,j),2) %></td>
<td><%=round(itemdifse(1,j),2)%></td>

<td> <%=round(itemdif(0,j)-itemdif(1,j),2)%></td>
<td> <%=round(see2,2)%></td>
<td><%=round(tvalue,2)%></td>
<td><%=tvalue2%></td> 
<td><%=df %></td><td><%=j %></td></tr> 
 
<%   
  next
 
   end if 'ubound(kgender)=1
 
%>
 </table>
<%

  if ubound(kgender)=1 then   
%>
 <table><tr>DIF class/group specification is: <font color=red>Pairwise DIF=<%=mgender%>(2)</font></tr>
 
<tr><td>  KID  </td><td>Obs-Exp</td><td>DIF </td><td>DIF</td><td>KID</td><td>Obs-Exp </td><td> DIF </td><td>DIF</td><td> DIF</td><td> JOINT</td><td>Rasch-</td><td>Welch</td><td>  Item</td><td> </td></tr>                               
<tr><td>  CLASS</td><td>Average</td><td>MEASURE</td><td> S.E.</td><td> CLASS</td><td> Average</td><td>MEASURE</td><td> S.E. </td><td>  CONTRAST </td><td>S.E.</td><td> t </td><td>sig.</td><td>df</td><td>  Name</td></tr>                         
 <% j=1
  ' response.write persondif(jm)
  for j=1 to itemno 
 
      
               tvalue2=""
              see2=round(sqr(itemdifse(0,j)^2+itemdifse(1,j)^2),2)
               tvalue=round((itemdif(1,j)-itemdif(0,j))/see2,2)        
 if persondif(0)<persondif(1) then
     df=persondif(0)-1
 else
     df=persondif(1)-1
 end if
if df="" then df=10
if tvalue="" then tvalue=0
if abs(tvalue)>5 then tvalue=5
if df>100 then df=100
 
   strSQL =  "Select * From ttest2  where tvalue=" & abs(round(tvalue,2))
   Set objRS = GetSQLRecordset(strSQL, "../News.mdb", "ttest2")
 
 if objrs.eof then
     tvalue2= " <font color=red>" & round(0.001,3) &"</font>"
 else
   df2=df
     if df>50 then df2=50
  dfdd="df" & df
     tprob=objrs(dfdd)
    if tprob*2<0.05 then
         tvalue2= " <font color=red>" & round(tprob*2,3) &"</font>"
   else
           tvalue2=round(tprob*2,3) 
    end if
 end if
  %>
<tr align=center><td><%=kgender(1)%></td>
<td><%=round(difraw(1,j)/persondif(1)-difexp(1,j)/persondif(1),2)%></td>
<td><%= round(itemdif(1,j),2) %></td>
<td><%=round(itemdifse(1,j),2)%></td>
<td><%=kgender(0)%></td>
<td><%=round(difraw(0,j)/persondif(0)-difexp(0,j)/persondif(0),2)%></td>
<td><%= round(itemdif(0,j),2) %></td>
<td><%=round(itemdifse(0,j),2)%></td>

<td> <%=round(itemdif(1,j)-itemdif(0,j),2)%></td>
<td> <%=round(see2,2)%></td>
<td><%=round(tvalue,2)%></td>
<td><%=tvalue2%></td> 
<td><%=df%></td><td><%=j%></td></tr> 
 
<%   
 
next
 
   end if 'ubound(kgender)=1

%>
 </table>
<%


  if ubound(kgender)>=1 then 
%>
 <table><tr>DIF class/group specification is: <font color=red>Global DIF=<%=mgender%></font?</tr>
 
<tr><td>  Class   </td><td>OBSERV</td><td>ATIONS </td><td>BASE</td><td>LINE</td><td>DIF </td><td> DIF </td><td>DIF</td><td> DIF</td><td> DIF</td><td></td><td>  Item</td><td>   </td></tr>                               
<tr><td>  CLASS</td><td>COUNT</td><td>AVERAGE</td><td> EXPECT</td><td> MEASURE</td><td> SCORE</td><td>MEASURE</td><td> SIZE </td><td> S.E. </td><td>  t </td><td> sig.</td><td>df</td><td>  Name</td></tr>                         
 <% j=1
  ' response.write persondif(jm)
   for j=1 to itemno 
 
    for jm=0 to ubound(kgender)
        
               tvalue2=""
               tvalue=round((item(j)-itemdif(0,j))/itemdifse(jm,j),2)
          '   if abs(tvalue)>=2 then tvalue2="sig"
 
     df=persondif(0)-1
 
if df="" then df=10
if tvalue="" then tvalue=0
if abs(tvalue)>5 then tvalue=5
if df>100 then df=100
tvalue=abs(round(tvalue,2))
   strSQL =  "Select * From ttest2  where tvalue=" & tvalue 
   Set objRS = GetSQLRecordset(strSQL, "../News.mdb", "ttest2")
 
   if objrs.eof then
  tvalue2= " <font color=red>" & round(0.001,3) &"</font>"
 else
     df2=df
     if df>50 then df2=50
  dfdd="df" & df
     tprob=objrs(dfdd)
    if tprob*2<0.05 then
         tvalue2= " <font color=red>" & round(tprob*2,3) &"</font>"
   else
           tvalue2=round(tprob*2,3) 
    end if
 end if
  %>
<tr align=center><td> <%=kgender(jm)%></td><td><%=persondif(jm)%></td><td><%=round(difraw(jm,j)/persondif(jm),2)%></td><td> <%=round(difexp(jm,j)/persondif(jm),2)%></td><td><%=round(item(j),2)%></td><td><%=round(difraw(jm,j)/persondif(jm)-difexp(jm,j)/persondif(jm),2)%></td><td><%= round(itemdif(jm,j),2) %></td><td><%=round(item(j)-itemdif(jm,j),2)%></td><td><%= round(itemdifse(jm,j),2) %></td><td><%= round((item(j)-itemdif(jm,j))/itemdifse(jm,j),2) %></td><td><%=tvalue2%></td><td><%=df %></td><td><%=j %></td></tr> 
 
<%   next 
     
  next
   end if 'ubound(kgender)>1
%>

 </table>
<table><tr>DIF class/group specification is: <font color=red>Chi-square-DIF=<%=mgender%></font></tr>

<tr align=center><td>   </td><td>SUMMARY DIF</td><td></td><td></td><td></td><td>  Item</td><td>   </td></tr>                               
<tr><td>  CLASSES</td><td>CHI-SQUARED</td><td>D.F.</td><td> sig.</td><td>No.</td><td>  Name</td></tr>                         
 <% j=1
      
  for j=1 to itemno
           tvalue=0
       for jm=0 to   ubound(kgender)
         difx="dif" & jm+1
         sex="se" & jm+1
                   tvalue=tvalue+ round((item(j)-itemdif(jm,j))*(item(j)-itemdif(jm,j))/item(j),2)  
             '  tvalue=tvalue+ round((item(j)-itemdif(jm,j))/itemdifse(jm,j)*(item(j)-itemdif(jm,j))/itemdifse(jm,j) ,2)
                             'Zsquare
          next 
          tvalue2=""
        ' if tvalue/(ubound(kgender))>3.814 then tvalue2="sig"

df=ubound(kgender)
if df="" then df=2
if tvalue="" then tvalue=0
if abs(tvalue)>5 then tvalue=15
if df>100 then df=100
tvalue=abs(round(tvalue,2))
     
            if  df >100 then
                df=100
             else
                  if tvalue>49 then
                     tvalue=round(tvalue/(tvalue/50),2)
                      df=round(df/(tvalue/50),2)
                  end if
       ca2="a" & ubound(kgender)
         strSQL =  "Select * From chiquest where a0=" & round(tvalue,1)  
               Set objRSst = GetSQLRecordset(strSQL, "../kpiall/statistics.mdb", "chiquest")
              if objrsst.eof then
                  pro=0
              else
                 if df>100 or df<0 then
                    pro=1  
                 else   
                   itema="A" &  df   
                   pro=round(objrsst(itema),3)
                 end if
              end if
           end if	
 
  dfdd="df" & df
 
    if pro <0.05 then
         tvalue2= " <font color=red>" & round(pro,3) &"</font>"
   else
           tvalue2=round(pro,3) 
    end if
 
    %>
<tr><td> <%=ubound(kgender)+1 %></td><td><%=round(tvalue,2)%></td><td><%=ubound(kgender)%></td><td><%=tvalue2%></td><td><%=j %></td><td><%=j %></td></tr> 
 
<% 
      
    next  


           '06 07
      response.end
  end if '6

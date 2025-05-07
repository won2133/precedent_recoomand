import React from 'react';
import Menubar from '../elements/menubar';
import { useState, useEffect } from 'react';
import { Form, Button, Container, Row, Col } from 'react-bootstrap';
import axios from "axios";
import Header from './header.js';
import { Link } from 'react-router-dom';
import LoadingModal from '../elements/LoadingModal';
function Recommand ({apiUrl}) {
  const [userInput, setUserInput] = useState(""); //사용자 입력 내용
  const [result, setResult] = useState({}); //판례 추천 결과- [일련 번호, 판결요지]*5
const [check, setCheck] = useState("0"); //상태(서버에 post하면 submit으로 바꿈)
// 사용자 입력 내용 화면에 출력, textarea높이 조절
  const onUiChange = (e) => {
    e.target.style.height = '0px';
    let scrollHeight = e.target.scrollHeight;
    let borderTop = e.target.style.borderTop;
    let borderBottom = e.target.style.borderBottom;
    e.target.style.height = (scrollHeight + borderTop + borderBottom) + "px";
    setUserInput(e.target.value);
  };
  const handleSubmit = (e) => {
    // submit을 할 때 페이지 자체가 새로고침이 되는 것을 막음
    e.preventDefault();
    setCheck('submit');
    const formData = new FormData();
    formData.append('input', userInput)
    setResult({});
    axios.post(apiUrl+'/pan', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
  }).then((res)=> setResult(res.data)) 
  };
    return(
      <div>
        <header id="header">
        <Header/>
          </header>
        <nav id="nav">&nbsp;</nav>
        <div id="section">
          <div style={{backgroundColor:'#EEEEEE', display:'flex', borderRadius: '10px 10px 10px 10px / 10px 10px 10px 10px', position: 'absolute', left:'50%', marginLeft: -550, width:500, height:700}}>
             <form style= {{fontSize:28, margin: 'auto'}} onSubmit={handleSubmit}>
             <textarea spellcheck="false" style={{fontSize:20, margin: 'auto'}} id='recommandInput' onChange={onUiChange}  name="input" cols="45" rows="1" value={userInput} placeholder="상황을 입력해주세요"></textarea>
             <input className='b' type="submit" value="결과 보기"/>
             </form>
          </div>
          {(typeof result.pan_list === 'undefined' ? (
              <>{((check === 'submit') && <LoadingModal/>)} //result가 없고 check가 submit이면 로딩 화면
                                                                   출력
              <div style={{backgroundColor:'#EEEEEE', display:'flex', borderRadius: '10px 10px 10px 10px / 10px 10px 10px 10px', position: 'absolute', left:'50%', marginLeft: 50, width:500, height:700, paddingTop:295, paddingLeft:227, fontSize:80}}>?
                </div></>) : (
                <div style={{ display:'inline-block', position: 'absolute', left:'50%', marginLeft: 50}}>
                  {result.pan_list.map((r, idx) => {
                    return (
                      <p className='block'>
                        <Link style={{textDecoration: "none" ,color:'black', margin: 'auto'}} to={'/precedent/detail/' + r[0] + '/' + 1}>
                        {r[1].map((s) => {
                            return (
                              <p style={{margin: 'auto', fontSize: '17px'}}>{s.slice(0, 130)}</p>
                            )})}
                        </Link>
                      </p>
                    )})}
                </div>
              ))}
        </div>
        <aside id="aside">&nbsp;</aside>
        <footer style={{marginTop:220}}>*저희 서비스는 참고용으로만 이용해주세요*</footer>
      </div>
    )}
export default Recommand;

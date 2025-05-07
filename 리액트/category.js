import { useState, useEffect } from "react";
import axios from 'axios'
import { Navbar, Container, Nav, NavDropdown } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import CategoryConts from "./category_conts.js;
import Menubar from '../elements/menubar';
import { useParams } from "react-router";
import Header from './header.js';
import LoadingModal from '../elements/LoadingModal';
//하위 카테고리- tabName은 상위 카테고리(str), kind는 민법/판례, c2는 하위 카테고리(int)
function TabContent({tabName, kind, c2}){
  const tabContList = {'총칙': ['통칙', '인', '법인', '물건', '법률행위', '기간', '소멸시효'],
  '물권': ['총칙', '점유권', '소유권', '지상권', '지역권', '전세권', '유치권', '질권', '저당권'],
  '채권' : ['총칙', '증여', '매매', '교환', '소비대차', '사용대차', '임대차', '고용', '도급', '여행계약',
  '현상광고', '위임', '임치', '조합', '종신정기금', '화해', '사무관리', '부당이득', '불법행위'],
  '친족' : ['총칙', '가족의 범위와 자의 성과 본', '혼인', '친생자', '양자', '친권', '후견', '부양'],
  '상속' : ['상속', '유언', '유류분']};
  return <Nav className="me-auto" style={{fontSize:12}}>{  
          tabContList[tabName].map((c, idx)=>{
            return (<Nav.Link onClick={()=>{ 
                      let url = '' //민법인지 판례인지에 따라 url(클릭 시 이동할 페이지) 지정
                      kind === 'article' ? (url = "/category/article" + "/" + tabName + "/" + idx) :
                      (url = "/category/precedent" + "/" + tabName + "/" + c)
                      window.location.href =url
                    }}>
                    {(c === c2 | String(idx) === c2) ? (
                      <span  className="fontFt" style={{color: '#333333', fontWeight:'bold', fontSize:17}}>{c}</span>) : 
                    (<span  className="fontFt" style={{fontSize:17}}>{c}</span>)}</Nav.Link>)})
      }
      </Nav>
}
//상위 카테고리- kind는 민법/판례, c1은 상위 카테고리(str), c2는 하위 카테고리(int)
function Tab({kind, c1, c2}) {
  let [tabName, setTabName] = useState(c1);
  const tabList = ['총칙', '물권', '채권', '친족', '상속'];
  return (
<div>
<Nav className="mt-5 mb-3" variant="tabs" defaultActiveKey={c1}>
      {tabList.map((t, idx)=> {
          let k = t;
          return (    
              <Nav.Item>
                <Nav.Link eventKey={k} onClick={()=>{setTabName(t);}}>
                    <span  style={{fontSize:18}}>{t}</span>
                 </Nav.Link>
              </Nav.Item>
          )})}
  </Nav>
  <Navbar bg="light" expand="xxl">
  <Container>
<Navbar.Brand href="#home">
<span className="fontTw" style={{color:'#000000'}}>{tabName}</span>
</Navbar.Brand>
  <Navbar.Toggle aria-controls="basic-navbar-nav"/>
  <Navbar.Collapse id="basic-navbar-nav">
    <TabContent tabName={tabName} kind={kind} c2={c2}/>
  </Navbar.Collapse></Container></Navbar>
  </div>
)}
export default function Category({kind, apiUrl}) {
  let [getData, setGetData] = useState(0);
  const {c1, c2} = useParams(); //c1은 상위 카테고리, c2는 하위 카테고리
//새로고침(이동)할 때마다 kind(민법/판례)와 c2에 맞는 내용 요청
  useEffect(() => {
    async function fetchData() {
      const result = await axios.get(
        apiUrl + '/' + kind + '/' + c1 +"/" + c2
      );
       setGetData(result.data);
    }
    fetchData()
  }, [])
  return (
    <div>
    <header id="header"><Header/></header>
    <nav id="nav">&nbsp;</nav>
    <div id="section">
    <div>
        {(typeof getData.c_list2 === 'undefined') ? (<LoadingModal/>) : (
(typeof getData.c_list1 === 'undefined') || (<div>
            <Tab c1={c1} c2={c2} kind={kind}/>
                <CategoryConts list={getData.dic.x_list} total={getData.dic.total} kind={kind}/>
            </div>
        ))}
    </div>
    </div>
    <aside id="aside">&nbsp;</aside>
    <footer></footer>
  </div>
)}

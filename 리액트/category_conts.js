import React from 'react';
import Header from './header.js';
import axios from 'axios'
import { useState, useEffect, useRef } from 'react';
import Pagination from "react-js-pagination";
import { Link } from 'react-router-dom';
import { AiOutlineStar } from "react-icons/ai";
import { AiFillStar } from "react-icons/ai";
function Category_conts({list, total, kind}) {
    let [curList, setCurList] = useState([]); //민법/판례 제목 리스트(최대 10개)
    const [page, setPage] = useState(1); //페이지 번호(디폴트 1)
    //새로고침(이동) 될 때 curList 설정(page=1이므로 최조 10개)
useEffect(() => {
        let new_l = [];
        let last_num = page*10;
        if (total < last_num) {
          last_num =  total;
        }
        setCurList(list.slice((page-1)*10, last_num))
    }, [])
      const Paging = () => {
        //page 바뀔 때마다 curList 새로 설정
        const handlePageChange = (page) => {
          setPage(page);
          let last_num = page*10;
          if (total < last_num) {
            last_num =  total;
          }
          setCurList(list.slice((page-1)*10, last_num))
        };
        return (
          <div style={{position:'absolute', left:'50%', marginLeft:'-70px'}}>
          <Pagination 
            style={{display:'block'}}
            activePage={page} // 현재 페이지
            itemsCountPerPage={10} // 한 페이지랑 보여줄 아이템 개수
            totalItemsCount={total} // 총 아이템 개수
            pageRangeDisplayed={5} // paginator의 페이지 범위
            prevPageText={"‹"} // "이전"을 나타낼 텍스트
            nextPageText={"›"} // "다음"을 나타낼 텍스트
            onChange={handlePageChange} // 페이지 변경을 핸들링하는 함수
          />
          </div>
        );
      };
    return (
        <div class="div2">
            <table class="table table-hover fontTw">
                 <thead><tr>
                    <th id="t1" ><span className='fontTw'>번호</span></th>
                    <th style={{width:900}}>{(kind==='article' ? (<span>내용</span>) : 
                        (<span>제목</span>))}</th>
                </tr></thead>
                 <tbody>
                   {curList.map((string, idx) => {
                     let url = "/precedent/detail/" + string[1] + '/' + page
                       return(
                         <tr>
                           <td className="widthTh"><span>{(page-1)*10+idx+1}</span></td>
                           <td>
                             {kind==='article' ? (
                               <details>
                                 <summary>{string[0]}</summary><br/>
                                 <p>{string[1].map((str)=> {return <p>{str}</p>})}</p></details>) : (
                               <span><Link to={url}  style={{ textDecoration: "none" , color:'black'}}>{string[2]}</Link></span>
                             )}
                           </td>
                         </tr>
                   )})}
                 </tbody>
            </table>
            <Paging/>
        </div>
    )}
export default Category_conts;
 
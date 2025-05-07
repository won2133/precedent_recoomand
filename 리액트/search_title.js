import React from 'react';
import Header from './header.js';
function Search_title({name, num}) {
    return (
        <div class = "div1">
        <p class="div1-title">{name}</p>
        <p class= 'div1-sub'>
            <ul>
            <li class="div1-sub1"></li>
            <li class="div1-sub2">&nbsp;&nbsp;&nbsp;&nbsp; (총 {num} 건)</li>
            </ul>
        </p>
        </div>
    );
}
export default Search_title;
